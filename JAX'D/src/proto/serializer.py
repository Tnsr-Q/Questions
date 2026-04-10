from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any, Dict, Type, Union

from .constraint_schema import PhysicsPredicate


class Serializer:
    """Serialization helpers for predicates and metadata."""

    SUPPORTED_FORMATS = ["json", "protobuf", "arrow", "pickle"]

    @staticmethod
    def serialize(
        obj: Union[PhysicsPredicate, Dict[str, Any]], format: str = "json", version: str = "1.0.0"
    ) -> Union[str, bytes]:
        _ = version

        if format == "json":
            if isinstance(obj, PhysicsPredicate):
                return obj.to_json()
            return json.dumps(obj)

        if format == "protobuf":
            if isinstance(obj, PhysicsPredicate):
                return obj.to_protobuf()
            return json.dumps(obj).encode("utf-8")

        if format == "arrow":
            import pyarrow as pa

            if not isinstance(obj, dict):
                raise ValueError("Arrow serialization requires dict input")
            table = pa.Table.from_pydict(obj)
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, table.schema) as writer:
                writer.write_table(table)
            return sink.getvalue().to_pybytes()

        if format == "pickle":
            return pickle.dumps(obj)

        raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def deserialize(
        data: Union[str, bytes], format: str = "json", target_cls: Type[Any] = PhysicsPredicate
    ) -> Any:
        if format == "json":
            raw = json.loads(data if isinstance(data, str) else data.decode("utf-8"))
            return PhysicsPredicate(**raw) if target_cls is PhysicsPredicate else raw

        if format == "protobuf":
            raw = json.loads(data.decode("utf-8"))
            return PhysicsPredicate(**raw) if target_cls is PhysicsPredicate else raw

        if format == "arrow":
            import pyarrow as pa

            with pa.ipc.open_stream(pa.py_buffer(data)) as reader:
                table = reader.read_all()
            return table.to_pydict()

        if format == "pickle":
            return pickle.loads(data)

        raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def compute_checksum(data: Union[str, bytes]) -> str:
        if isinstance(data, str):
            data = data.encode("utf-8")
        return hashlib.sha256(data).hexdigest()
