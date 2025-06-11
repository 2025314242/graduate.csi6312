from typing import Any, Dict


def serialize_table(
    table: Dict[str, Any],
    is_title: bool=True,
    is_header: bool=True,
    is_cell: bool=True
    ) -> str:
    """Serialize table

    [Params]
    table     : Dict[str, Any]
    is_title  : bool
    is_header : bool
    is_cell   : bool

    [Return]
    serialized_table : str
    """
    serialized_table_content = ["Table"]
    serialized_table_content.append(f"[title] {table['title']}" if is_title else "")
    serialized_table_content.append(f"[header] {' | '.join(table['header'])}" if is_header else "")
    serialized_table_content.append(
        ' '.join(
            [f"[row {row_idx + 1}] {' | '.join(row)}" for row_idx, row in enumerate(table['cell'])]
        )
        if is_cell else ""
    )

    serialized_table = ' '.join([content for content in serialized_table_content if content != ""])
    
    return serialized_table
