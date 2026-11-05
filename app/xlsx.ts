// 零依赖的 .xlsx 生成器：ZIP（store，不压缩）+ 最小 OOXML 部件。
// 支持多工作表、行内字符串、数字、列宽、可选冻结首行，以及一套统一样式
// （表头底色、区块标题、键/值区分、合计行、千分位与两位小数）。
// 生成的是真正的 OOXML 工作簿，Excel / Numbers / LibreOffice 均可直接打开。

export type Cell = string | number | null | undefined;

/** 内置样式：见 stylesXml 里的 cellXfs 顺序。 */
export type CellStyle =
  | "default"
  | "header"
  | "section"
  | "label"
  | "value"
  | "totalLabel"
  | "totalBytes"
  | "totalDecimal"
  | "bytes"
  | "decimal";

const STYLE_INDEX: Record<CellStyle, number> = {
  default: 0,
  header: 1,
  section: 2,
  label: 3,
  value: 4,
  totalLabel: 5,
  totalBytes: 6,
  totalDecimal: 7,
  bytes: 8,
  decimal: 9,
};

export type SheetSpec = {
  /** 工作表名（≤31 字符，不能含 : \ / ? * [ ]） */
  name: string;
  /** 每列的字符宽度，缺省 18 */
  columnWidths?: number[];
  /** 行数据；第一行视为表头（自动套用 header 样式） */
  rows: Cell[][];
  /** 按 (行, 列, 值) 返回单元格样式；表头行默认 header */
  cellStyle?: (rowIndex: number, columnIndex: number, value: Cell) => CellStyle;
  /** 是否冻结首行（默认不冻结） */
  freezeHeader?: boolean;
};

const XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main";
const REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships";

/** XML 文本转义。 */
function escapeXml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

/** 0 → A、25 → Z、26 → AA。 */
function columnName(index: number): string {
  let n = index + 1;
  let name = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    name = String.fromCharCode(65 + rem) + name;
    n = Math.floor((n - 1) / 26);
  }
  return name;
}

function cellXml(ref: string, value: Cell, styleIndex: number): string {
  const style = styleIndex > 0 ? ` s="${styleIndex}"` : "";
  if (value === null || value === undefined || value === "") {
    return styleIndex > 0 ? `<c r="${ref}"${style}/>` : "";
  }
  if (typeof value === "number" && Number.isFinite(value)) {
    return `<c r="${ref}"${style}><v>${value}</v></c>`;
  }
  return `<c r="${ref}"${style} t="inlineStr"><is><t xml:space="preserve">${escapeXml(String(value))}</t></is></c>`;
}

function sheetXml(sheet: SheetSpec): string {
  const rows = sheet.rows.length > 0 ? sheet.rows : [[""]];
  const columnCount = rows.reduce((max, row) => Math.max(max, row.length), 1);
  const widths = sheet.columnWidths ?? [];
  const cols = Array.from({ length: columnCount }, (_, i) =>
    `<col min="${i + 1}" max="${i + 1}" width="${widths[i] ?? 18}" customWidth="1"/>`,
  ).join("");
  const styleOf = sheet.cellStyle;
  const body = rows
    .map((row, rowIndex) => {
      const cells = row
        .map((value, columnIndex) => {
          const styleId = styleOf
            ? styleOf(rowIndex, columnIndex, value)
            : rowIndex === 0
              ? "header"
              : "default";
          return cellXml(`${columnName(columnIndex)}${rowIndex + 1}`, value, STYLE_INDEX[styleId]);
        })
        .join("");
      return `<row r="${rowIndex + 1}">${cells}</row>`;
    })
    .join("");
  const pane = sheet.freezeHeader
    ? '<pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>'
    : "";
  return (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    `<worksheet xmlns="${XLSX_NS}">` +
    `<sheetViews><sheetView workbookViewId="0">${pane}</sheetView></sheetViews>` +
    '<sheetFormatPr defaultRowHeight="15"/>' +
    `<cols>${cols}</cols>` +
    `<sheetData>${body}</sheetData>` +
    "</worksheet>"
  );
}

function workbookXml(sheets: SheetSpec[]): string {
  const entries = sheets
    .map((sheet, index) => `<sheet name="${escapeXml(sheet.name)}" sheetId="${index + 1}" r:id="rId${index + 1}"/>`)
    .join("");
  return (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    `<workbook xmlns="${XLSX_NS}" xmlns:r="${REL_NS}">` +
    `<sheets>${entries}</sheets>` +
    "</workbook>"
  );
}

function workbookRelsXml(sheets: SheetSpec[]): string {
  const sheetRels = sheets
    .map(
      (_, index) =>
        `<Relationship Id="rId${index + 1}" Type="${REL_NS}/worksheet" Target="worksheets/sheet${index + 1}.xml"/>`,
    )
    .join("");
  return (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    `<Relationships xmlns="${PKG_REL_NS}">` +
    sheetRels +
    `<Relationship Id="rId${sheets.length + 1}" Type="${REL_NS}/styles" Target="styles.xml"/>` +
    "</Relationships>"
  );
}

function contentTypesXml(sheets: SheetSpec[]): string {
  const overrides = sheets
    .map(
      (_, index) =>
        `<Override PartName="/xl/worksheets/sheet${index + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>`,
    )
    .join("");
  return (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">' +
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>' +
    '<Default Extension="xml" ContentType="application/xml"/>' +
    '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>' +
    overrides +
    '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>' +
    "</Types>"
  );
}

function rootRelsXml(): string {
  return (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    `<Relationships xmlns="${PKG_REL_NS}">` +
    `<Relationship Id="rId1" Type="${REL_NS}/officeDocument" Target="xl/workbook.xml"/>` +
    "</Relationships>"
  );
}

/**
 * 样式表：0 普通 / 1 表头（深色底白字居中）/ 2 区块标题 / 3 键名 / 4 值 /
 * 5 合计文字 / 6 合计字节（千分位）/ 7 合计小数 / 8 字节（千分位）/ 9 小数（两位）。
 */
function stylesXml(): string {
  const xf = (
    index: number,
    fontId: number,
    fillId: number,
    borderId: number,
    numFmtId = 0,
  ): string => {
    const attrs = [`numFmtId="${numFmtId}"`, `fontId="${fontId}"`, `fillId="${fillId}"`, `borderId="${borderId}"`, 'xfId="0"'];
    if (numFmtId !== 0) attrs.push('applyNumberFormat="1"');
    if (fontId !== 0) attrs.push('applyFont="1"');
    if (fillId !== 0) attrs.push('applyFill="1"');
    if (borderId !== 0) attrs.push('applyBorder="1"');
    if (index === 1) {
      attrs.push('applyAlignment="1"');
      return `<xf ${attrs.join(" ")}><alignment horizontal="center" vertical="center"/></xf>`;
    }
    return `<xf ${attrs.join(" ")}/>`;
  };

  return (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>' +
    `<styleSheet xmlns="${XLSX_NS}">` +
    '<fonts count="3">' +
    '<font><sz val="11"/><color rgb="FF24322C"/><name val="Calibri"/></font>' +
    '<font><b/><sz val="11"/><color rgb="FFFFFFFF"/><name val="Calibri"/></font>' +
    '<font><b/><sz val="11"/><color rgb="FF1F3A32"/><name val="Calibri"/></font>' +
    "</fonts>" +
    '<fills count="5">' +
    '<fill><patternFill patternType="none"/></fill>' +
    '<fill><patternFill patternType="gray125"/></fill>' +
    '<fill><patternFill patternType="solid"><fgColor rgb="FF2F5D50"/><bgColor indexed="64"/></patternFill></fill>' +
    '<fill><patternFill patternType="solid"><fgColor rgb="FFE8F1EC"/><bgColor indexed="64"/></patternFill></fill>' +
    '<fill><patternFill patternType="solid"><fgColor rgb="FFF3F6F5"/><bgColor indexed="64"/></patternFill></fill>' +
    "</fills>" +
    '<borders count="2">' +
    "<border/>" +
    '<border><left style="thin"><color rgb="FFC9D6D0"/></left><right style="thin"><color rgb="FFC9D6D0"/></right><top style="thin"><color rgb="FFC9D6D0"/></top><bottom style="thin"><color rgb="FFC9D6D0"/></bottom><diagonal/></border>' +
    "</borders>" +
    '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>' +
    '<cellXfs count="10">' +
    [xf(0, 0, 0, 0), xf(1, 1, 2, 1), xf(2, 2, 3, 1), xf(3, 2, 4, 1), xf(4, 0, 0, 1), xf(5, 2, 3, 1), xf(6, 2, 3, 1, 3), xf(7, 2, 3, 1, 2), xf(8, 0, 0, 1, 3), xf(9, 0, 0, 1, 2)].join("") +
    "</cellXfs>" +
    "</styleSheet>"
  );
}

// ---------- ZIP（store 模式） ----------

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i += 1) {
    let c = i;
    for (let k = 0; k < 8; k += 1) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c >>> 0;
  }
  return table;
})();

function crc32(bytes: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < bytes.length; i += 1) {
    crc = CRC_TABLE[(crc ^ bytes[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

type ZipEntry = { path: string; data: Uint8Array };

function zip(entries: ZipEntry[]): Uint8Array {
  const encoder = new TextEncoder();
  const chunks: Uint8Array[] = [];
  const central: Uint8Array[] = [];
  let offset = 0;

  for (const entry of entries) {
    const nameBytes = encoder.encode(entry.path);
    const crc = crc32(entry.data);
    const size = entry.data.length;

    const local = new Uint8Array(30 + nameBytes.length);
    const lv = new DataView(local.buffer);
    lv.setUint32(0, 0x04034b50, true);
    lv.setUint16(4, 20, true); // version needed
    lv.setUint16(6, 0x0800, true); // UTF-8 文件名
    lv.setUint16(8, 0, true); // store
    lv.setUint16(10, 0, true); // time
    lv.setUint16(12, 0x21, true); // date（1980-01-01）
    lv.setUint32(14, crc, true);
    lv.setUint32(18, size, true);
    lv.setUint32(22, size, true);
    lv.setUint16(26, nameBytes.length, true);
    lv.setUint16(28, 0, true);
    local.set(nameBytes, 30);

    chunks.push(local, entry.data);

    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const cv = new DataView(centralHeader.buffer);
    cv.setUint32(0, 0x02014b50, true);
    cv.setUint16(4, 20, true); // version made by
    cv.setUint16(6, 20, true); // version needed
    cv.setUint16(8, 0x0800, true);
    cv.setUint16(10, 0, true);
    cv.setUint16(12, 0, true);
    cv.setUint16(14, 0x21, true);
    cv.setUint32(16, crc, true);
    cv.setUint32(20, size, true);
    cv.setUint32(24, size, true);
    cv.setUint16(28, nameBytes.length, true);
    cv.setUint16(30, 0, true); // extra
    cv.setUint16(32, 0, true); // comment
    cv.setUint16(34, 0, true); // disk
    cv.setUint16(36, 0, true); // internal attrs
    cv.setUint32(38, 0, true); // external attrs
    cv.setUint32(42, offset, true);
    centralHeader.set(nameBytes, 46);
    central.push(centralHeader);

    offset += local.length + size;
  }

  const centralSize = central.reduce((sum, item) => sum + item.length, 0);
  const eocd = new Uint8Array(22);
  const ev = new DataView(eocd.buffer);
  ev.setUint32(0, 0x06054b50, true);
  ev.setUint16(4, 0, true);
  ev.setUint16(6, 0, true);
  ev.setUint16(8, entries.length, true);
  ev.setUint16(10, entries.length, true);
  ev.setUint32(12, centralSize, true);
  ev.setUint32(16, offset, true);
  ev.setUint16(20, 0, true);

  const total = offset + centralSize + eocd.length;
  const out = new Uint8Array(total);
  let cursor = 0;
  for (const chunk of [...chunks, ...central, eocd]) {
    out.set(chunk, cursor);
    cursor += chunk.length;
  }
  return out;
}

/** 生成 .xlsx 字节流。 */
export function buildXlsx(sheets: SheetSpec[]): Uint8Array {
  if (sheets.length === 0) throw new Error("至少需要一个工作表");
  const encoder = new TextEncoder();
  const entries: ZipEntry[] = [
    { path: "[Content_Types].xml", data: encoder.encode(contentTypesXml(sheets)) },
    { path: "_rels/.rels", data: encoder.encode(rootRelsXml()) },
    { path: "xl/workbook.xml", data: encoder.encode(workbookXml(sheets)) },
    { path: "xl/_rels/workbook.xml.rels", data: encoder.encode(workbookRelsXml(sheets)) },
    { path: "xl/styles.xml", data: encoder.encode(stylesXml()) },
    ...sheets.map((sheet, index) => ({
      path: `xl/worksheets/sheet${index + 1}.xml`,
      data: encoder.encode(sheetXml(sheet)),
    })),
  ];
  return zip(entries);
}

/** 浏览器侧：生成工作簿并触发下载（服务端渲染环境不要调用）。 */
export function downloadXlsx(sheets: SheetSpec[], fileName: string): void {
  const bytes = buildXlsx(sheets);
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
  const blob = new Blob([buffer], {
    type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}
