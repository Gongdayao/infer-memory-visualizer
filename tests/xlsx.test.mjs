import assert from "node:assert/strict";
import test from "node:test";

import { buildXlsx } from "../app/xlsx.ts";

function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let k = 0; k < 8; k += 1) crc = crc & 1 ? 0xedb88320 ^ (crc >>> 1) : crc >>> 1;
  }
  return (crc ^ 0xffffffff) >>> 0;
}

/** 读取 store 模式的 ZIP：返回 { path: Uint8Array }，并顺带校验 CRC。 */
function readZip(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const decoder = new TextDecoder();
  let eocd = -1;
  for (let i = bytes.length - 22; i >= 0; i -= 1) {
    if (view.getUint32(i, true) === 0x06054b50) {
      eocd = i;
      break;
    }
  }
  assert.notEqual(eocd, -1, "应存在 ZIP 结尾记录");
  const count = view.getUint16(eocd + 10, true);
  let cursor = view.getUint32(eocd + 16, true);
  const entries = {};
  for (let index = 0; index < count; index += 1) {
    assert.equal(view.getUint32(cursor, true), 0x02014b50, "中央目录签名");
    const crc = view.getUint32(cursor + 16, true);
    const size = view.getUint32(cursor + 24, true);
    const nameLength = view.getUint16(cursor + 28, true);
    const extraLength = view.getUint16(cursor + 30, true);
    const commentLength = view.getUint16(cursor + 32, true);
    const localOffset = view.getUint32(cursor + 42, true);
    const name = decoder.decode(bytes.subarray(cursor + 46, cursor + 46 + nameLength));

    assert.equal(view.getUint32(localOffset, true), 0x04034b50, "本地文件头签名");
    const localNameLength = view.getUint16(localOffset + 26, true);
    const localExtraLength = view.getUint16(localOffset + 28, true);
    const dataStart = localOffset + 30 + localNameLength + localExtraLength;
    const data = bytes.subarray(dataStart, dataStart + size);
    assert.equal(crc32(data), crc, `${name} 的 CRC 应匹配`);
    entries[name] = data;
    cursor += 46 + nameLength + extraLength + commentLength;
  }
  return entries;
}

const decoder = new TextDecoder();

test("生成的 xlsx 是合法 ZIP，包含全部 OOXML 部件", () => {
  const bytes = buildXlsx([
    { name: "概要", rows: [["项目", "值"], ["模型", "Step-5-Preview"]] },
    { name: "显存构成", rows: [["项目", "字节数"], ["单卡合计", 18427131904]] },
  ]);
  assert.ok(bytes.length > 0);
  const entries = readZip(bytes);
  assert.deepEqual(Object.keys(entries).sort(), [
    "[Content_Types].xml",
    "_rels/.rels",
    "xl/_rels/workbook.xml.rels",
    "xl/styles.xml",
    "xl/workbook.xml",
    "xl/worksheets/sheet1.xml",
    "xl/worksheets/sheet2.xml",
  ]);

  const workbook = decoder.decode(entries["xl/workbook.xml"]);
  assert.match(workbook, /<sheet name="概要" sheetId="1" r:id="rId1"\/>/);
  assert.match(workbook, /<sheet name="显存构成" sheetId="2" r:id="rId2"\/>/);

  const types = decoder.decode(entries["[Content_Types].xml"]);
  assert.match(types, /PartName="\/xl\/worksheets\/sheet2\.xml"/);
  assert.match(types, /PartName="\/xl\/styles\.xml"/);

  const rels = decoder.decode(entries["xl/_rels/workbook.xml.rels"]);
  assert.match(rels, /Target="worksheets\/sheet1\.xml"/);
  assert.match(rels, /Target="styles\.xml"/);
});

test("单元格：数字写成数值、文本写成 inlineStr、特殊字符转义、表头套用样式", () => {
  const bytes = buildXlsx([
    {
      name: "明细",
      columnWidths: [30, 12],
      rows: [
        ["项目", "数值"],
        ["单卡总显存", 18427131904],
        ["备注", '含 <特殊> & "引号"'],
        ["空值", null],
      ],
    },
  ]);
  const sheet = decoder.decode(readZip(bytes)["xl/worksheets/sheet1.xml"]);

  assert.doesNotMatch(sheet, /<pane/, "默认不冻结首行");
  assert.match(sheet, /<col min="1" max="1" width="30" customWidth="1"\/>/, "列宽应写入");
  assert.match(sheet, /<c r="A1" s="1" t="inlineStr"><is><t xml:space="preserve">项目<\/t><\/is><\/c>/, "表头套用 header 样式");
  assert.match(sheet, /<c r="B2"><v>18427131904<\/v><\/c>/, "数字应为数值单元格");
  assert.match(sheet, /含 &lt;特殊&gt; &amp; &quot;引号&quot;/, "文本应转义");
  assert.doesNotMatch(sheet, /<c r="B4"/, "默认样式下空值不写单元格");
});

test("cellStyle 可为任意单元格指定样式，freezeHeader 可选开启", () => {
  const bytes = buildXlsx([
    {
      name: "样式",
      freezeHeader: true,
      rows: [["表头", "值"], ["标签", 12], ["合计", 34]],
      cellStyle: (row, col) =>
        row === 0 ? "header" : row === 2 ? (col === 0 ? "totalLabel" : "totalBytes") : col === 0 ? "label" : "value",
    },
  ]);
  const sheet = decoder.decode(readZip(bytes)["xl/worksheets/sheet1.xml"]);
  assert.match(sheet, /<pane ySplit="1" topLeftCell="A2"/, "freezeHeader 开启时冻结");
  assert.match(sheet, /<c r="A1" s="1"/, "表头 header 样式");
  assert.match(sheet, /<c r="A2" s="3"/, "键名 label 样式");
  assert.match(sheet, /<c r="B2" s="4"><v>12<\/v><\/c>/, "值 value 样式");
  assert.match(sheet, /<c r="A3" s="5"/, "合计文字 totalLabel 样式");
  assert.match(sheet, /<c r="B3" s="6"><v>34<\/v><\/c>/, "合计字节 totalBytes 样式（千分位）");
});

test("样式表包含表头底色、边框与十进制格式", () => {
  const bytes = buildXlsx([{ name: "样式", rows: [["a", 1]] }]);
  const styles = decoder.decode(readZip(bytes)["xl/styles.xml"]);
  assert.match(styles, /<fonts count="3">/);
  assert.match(styles, /<fills count="5">/);
  assert.match(styles, /<fgColor rgb="FF2F5D50"/, "表头深色底");
  assert.match(styles, /<border><left style="thin"/, "细边框");
  assert.match(styles, /<cellXfs count="10">/);
  assert.match(styles, /numFmtId="3"/, "千分位数字格式");
  assert.match(styles, /numFmtId="2"/, "两位小数格式");
});

test("空工作簿应报错，单表也能生成", () => {
  assert.throws(() => buildXlsx([]), /至少需要一个工作表/);
  const bytes = buildXlsx([{ name: "只有一行", rows: [["a", "b"]] }]);
  const entries = readZip(bytes);
  assert.equal(Object.keys(entries).length, 6);
  assert.match(decoder.decode(entries["xl/worksheets/sheet1.xml"]), /<c r="B1" s="1"/);
});
