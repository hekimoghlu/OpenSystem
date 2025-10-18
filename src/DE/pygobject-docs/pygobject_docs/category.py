import logging
import types
from enum import IntEnum, IntFlag, StrEnum, auto

from gi.module import repository
from gi._gi import EnumInfo, FunctionInfo, GType, InterfaceInfo, ObjectInfo, StructInfo, UnionInfo, VFuncInfo
from gi.types import GObjectMeta, StructMeta
from gi.repository import GObject


log = logging.getLogger(__name__)


class Category(StrEnum):
    Functions = auto()
    Interfaces = auto()
    Classes = auto()
    Structures = auto()  # GI type: record
    ClassStructures = auto()
    Unions = auto()
    Enums = auto()
    Constants = auto()
    Ignored = auto()

    @property
    def single(self):
        if self is Category.Classes:
            return "class"
        return str(self)[:-1]


class MemberCategory(StrEnum):
    Constructors = auto()
    Methods = auto()
    Properties = auto()
    Fields = auto()
    Signals = auto()
    VirtualMethods = auto()
    Ignored = auto()


def determine_category(module, name, gir=None) -> Category:
    """Determine the category to put the field in

    The category is based on the GI type info. For custom
    and overridden fields some extra checks are done.
    """
    try:
        field = getattr(module, name)
    except RuntimeError:
        log.warning("Failed to get field %s.%s. Ignoring it for now.", module, name)
        return Category.Ignored

    namespace = module.__name__.split(".")[-1]
    info = repository.find_by_name(namespace, name)

    if name.startswith("_") or isinstance(field, types.ModuleType):
        return Category.Ignored
    elif isinstance(info, FunctionInfo) or isinstance(
        field,
        (
            FunctionInfo,
            types.FunctionType,
            types.BuiltinFunctionType,
        ),
    ):
        return Category.Functions
    elif isinstance(info, UnionInfo):
        return Category.Unions
    elif isinstance(info, EnumInfo):
        return Category.Enums
    elif isinstance(info, StructInfo) or isinstance(field, StructMeta):
        if name.endswith("Private"):
            return Category.Ignored
        if gir and gir.struct_for(name):
            return Category.ClassStructures
        return Category.Structures
    elif isinstance(info, InterfaceInfo) or (namespace, name) == ("GObject", "GInterface"):
        return Category.Interfaces
    elif isinstance(info, ObjectInfo) or isinstance(field, (type, GObjectMeta)):
        return Category.Classes
    elif field is None or isinstance(field, (str, int, bool, float, tuple, dict, GType)):
        return Category.Constants

    raise TypeError(f"Type not recognized for {module.__name__}.{name}")


def determine_member_category(obj_type, name) -> MemberCategory:
    field = getattr(obj_type, name, None)

    if (
        name == "props"
        or name.startswith("_")
        or field in (GObject.Object._unsupported_method, GObject.Object._unsupported_data_method)
        or isinstance(field, type)
    ):
        return MemberCategory.Ignored
    elif isinstance(field, (FunctionInfo, types.MethodType)) and hasattr(field, "is_constructor"):
        return MemberCategory.Constructors if field.is_constructor() else MemberCategory.Methods
    elif isinstance(
        field,
        (
            types.FunctionType,
            types.BuiltinFunctionType,
            types.MethodType,
            types.MethodDescriptorType,
        ),
    ):
        return MemberCategory.Methods
    elif (
        isinstance(field, VFuncInfo)
        or field is None
        and name.startswith("do_")
        and name[3:] in (v.get_name() for v in obj_type.__info__.get_vfuncs())
    ):
        return MemberCategory.VirtualMethods
    elif isinstance(
        field,
        (
            GObject.GEnum,
            GObject.GFlags,
            GObject.Property,
            property,
            IntEnum,
            IntFlag,
            types.GetSetDescriptorType,
        ),
    ):
        return MemberCategory.Fields

    raise TypeError(f"Member type not recognized for {obj_type.__name__}.{name} ({getattr(obj_type, name)})")
