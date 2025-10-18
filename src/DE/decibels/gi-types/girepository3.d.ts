/**
 * GIRepository 3.0
 *
 * Generated from 3.0
 */

import * as GObject from "gobject2";
import * as GLib from "glib2";

export const TYPE_TAG_N_TYPES: number;
export function invoke_error_quark(): GLib.Quark;
export function type_tag_argument_from_hash_pointer(storage_type: TypeTag, hash_pointer: any | null): Argument;
export function type_tag_hash_pointer_from_argument(storage_type: TypeTag, arg: Argument): any | null;
export function type_tag_to_string(type: TypeTag): string;

export namespace ArrayType {
    export const $gtype: GObject.GType<ArrayType>;
}

export enum ArrayType {
    C = 0,
    ARRAY = 1,
    PTR_ARRAY = 2,
    BYTE_ARRAY = 3,
}

export namespace Direction {
    export const $gtype: GObject.GType<Direction>;
}

export enum Direction {
    IN = 0,
    OUT = 1,
    INOUT = 2,
}

export class InvokeError extends GLib.Error {
    static $gtype: GObject.GType<InvokeError>;

    constructor(options: { message: string; code: number });
    constructor(copy: InvokeError);

    // Fields
    static FAILED: number;
    static SYMBOL_NOT_FOUND: number;
    static ARGUMENT_MISMATCH: number;
}

export namespace RepositoryError {
    export const $gtype: GObject.GType<RepositoryError>;
}

export enum RepositoryError {
    TYPELIB_NOT_FOUND = 0,
    NAMESPACE_MISMATCH = 1,
    NAMESPACE_VERSION_CONFLICT = 2,
    LIBRARY_NOT_FOUND = 3,
}

export namespace ScopeType {
    export const $gtype: GObject.GType<ScopeType>;
}

export enum ScopeType {
    INVALID = 0,
    CALL = 1,
    ASYNC = 2,
    NOTIFIED = 3,
    FOREVER = 4,
}

export namespace Transfer {
    export const $gtype: GObject.GType<Transfer>;
}

export enum Transfer {
    NOTHING = 0,
    CONTAINER = 1,
    EVERYTHING = 2,
}

export namespace TypeTag {
    export const $gtype: GObject.GType<TypeTag>;
}

export enum TypeTag {
    VOID = 0,
    BOOLEAN = 1,
    INT8 = 2,
    UINT8 = 3,
    INT16 = 4,
    UINT16 = 5,
    INT32 = 6,
    UINT32 = 7,
    INT64 = 8,
    UINT64 = 9,
    FLOAT = 10,
    DOUBLE = 11,
    GTYPE = 12,
    UTF8 = 13,
    FILENAME = 14,
    ARRAY = 15,
    INTERFACE = 16,
    GLIST = 17,
    GSLIST = 18,
    GHASH = 19,
    ERROR = 20,
    UNICHAR = 21,
}

export namespace FieldInfoFlags {
    export const $gtype: GObject.GType<FieldInfoFlags>;
}

export enum FieldInfoFlags {
    READABLE = 1,
    WRITABLE = 2,
}

export namespace FunctionInfoFlags {
    export const $gtype: GObject.GType<FunctionInfoFlags>;
}

export enum FunctionInfoFlags {
    IS_METHOD = 1,
    IS_CONSTRUCTOR = 2,
    IS_GETTER = 4,
    IS_SETTER = 8,
    WRAPS_VFUNC = 16,
}

export namespace RepositoryLoadFlags {
    export const $gtype: GObject.GType<RepositoryLoadFlags>;
}

export enum RepositoryLoadFlags {
    NONE = 0,
    LAZY = 1,
}

export namespace VFuncInfoFlags {
    export const $gtype: GObject.GType<VFuncInfoFlags>;
}

export enum VFuncInfoFlags {
    CHAIN_UP = 1,
    OVERRIDE = 2,
    NOT_OVERRIDE = 4,
}
export module ArgInfo {
    export interface ConstructorProperties extends BaseInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class ArgInfo extends BaseInfo {
    static $gtype: GObject.GType<ArgInfo>;

    constructor(properties?: Partial<ArgInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<ArgInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    get_closure_index(): [boolean, number];
    get_destroy_index(): [boolean, number];
    get_direction(): Direction;
    get_ownership_transfer(): Transfer;
    get_scope(): ScopeType;
    get_type_info(): TypeInfo;
    is_caller_allocates(): boolean;
    is_optional(): boolean;
    is_return_value(): boolean;
    is_skip(): boolean;
    load_type_info(): TypeInfo;
    may_be_null(): boolean;
}
export module BaseInfo {
    export interface ConstructorProperties {
        [key: string]: any;
    }
}
export abstract class BaseInfo {
    static $gtype: GObject.GType<BaseInfo>;

    constructor(properties?: Partial<BaseInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<BaseInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    clear(): void;
    equal(info2: BaseInfo): boolean;
    get_attribute(name: string): string | null;
    get_container(): BaseInfo;
    get_name(): string | null;
    get_namespace(): string;
    get_typelib(): Typelib;
    is_deprecated(): boolean;
    iterate_attributes(iterator: AttributeIter): [boolean, AttributeIter, string, string];
    ref(): BaseInfo;
    unref(): void;
}
export module CallableInfo {
    export interface ConstructorProperties extends BaseInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export abstract class CallableInfo extends BaseInfo {
    static $gtype: GObject.GType<CallableInfo>;

    constructor(properties?: Partial<CallableInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<CallableInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    can_throw_gerror(): boolean;
    get_arg(n: number): ArgInfo;
    get_caller_owns(): Transfer;
    get_instance_ownership_transfer(): Transfer;
    get_n_args(): number;
    get_return_attribute(name: string): string | null;
    get_return_type(): TypeInfo;
    invoke(_function: any | null, in_args: Argument[], out_args: Argument[]): [boolean, Argument];
    is_method(): boolean;
    iterate_return_attributes(iterator: AttributeIter): [boolean, AttributeIter, string, string];
    load_arg(n: number): ArgInfo;
    load_return_type(): TypeInfo;
    may_return_null(): boolean;
    skip_return(): boolean;
}
export module CallbackInfo {
    export interface ConstructorProperties extends CallableInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class CallbackInfo extends CallableInfo {
    static $gtype: GObject.GType<CallbackInfo>;

    constructor(properties?: Partial<CallbackInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<CallbackInfo.ConstructorProperties>, ...args: any[]): void;
}
export module ConstantInfo {
    export interface ConstructorProperties extends BaseInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class ConstantInfo extends BaseInfo {
    static $gtype: GObject.GType<ConstantInfo>;

    constructor(properties?: Partial<ConstantInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<ConstantInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    get_type_info(): TypeInfo;
}
export module EnumInfo {
    export interface ConstructorProperties extends RegisteredTypeInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class EnumInfo extends RegisteredTypeInfo {
    static $gtype: GObject.GType<EnumInfo>;

    constructor(properties?: Partial<EnumInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<EnumInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    get_error_domain(): string | null;
    get_method(n: number): FunctionInfo;
    get_n_methods(): number;
    get_n_values(): number;
    get_storage_type(): TypeTag;
    get_value(n: number): ValueInfo;
}
export module FieldInfo {
    export interface ConstructorProperties extends BaseInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class FieldInfo extends BaseInfo {
    static $gtype: GObject.GType<FieldInfo>;

    constructor(properties?: Partial<FieldInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<FieldInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    get_flags(): FieldInfoFlags;
    get_offset(): number;
    get_size(): number;
    get_type_info(): TypeInfo;
}
export module FlagsInfo {
    export interface ConstructorProperties extends EnumInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class FlagsInfo extends EnumInfo {
    static $gtype: GObject.GType<FlagsInfo>;

    constructor(properties?: Partial<FlagsInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<FlagsInfo.ConstructorProperties>, ...args: any[]): void;
}
export module FunctionInfo {
    export interface ConstructorProperties extends CallableInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class FunctionInfo extends CallableInfo {
    static $gtype: GObject.GType<FunctionInfo>;

    constructor(properties?: Partial<FunctionInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<FunctionInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    get_flags(): FunctionInfoFlags;
    get_property(): PropertyInfo | null;
    get_symbol(): string;
    get_vfunc(): VFuncInfo | null;
}
export module InterfaceInfo {
    export interface ConstructorProperties extends RegisteredTypeInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class InterfaceInfo extends RegisteredTypeInfo {
    static $gtype: GObject.GType<InterfaceInfo>;

    constructor(properties?: Partial<InterfaceInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<InterfaceInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    find_method(name: string): FunctionInfo | null;
    find_signal(name: string): SignalInfo | null;
    find_vfunc(name: string): VFuncInfo | null;
    get_constant(n: number): ConstantInfo;
    get_iface_struct(): StructInfo | null;
    get_method(n: number): FunctionInfo;
    get_n_constants(): number;
    get_n_methods(): number;
    get_n_prerequisites(): number;
    get_n_properties(): number;
    get_n_signals(): number;
    get_n_vfuncs(): number;
    get_prerequisite(n: number): BaseInfo;
    get_property(n: number): PropertyInfo;
    get_signal(n: number): SignalInfo;
    get_vfunc(n: number): VFuncInfo;
}
export module ObjectInfo {
    export interface ConstructorProperties extends RegisteredTypeInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class ObjectInfo extends RegisteredTypeInfo {
    static $gtype: GObject.GType<ObjectInfo>;

    constructor(properties?: Partial<ObjectInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<ObjectInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    find_method(name: string): FunctionInfo | null;
    find_method_using_interfaces(name: string): [FunctionInfo | null, BaseInfo | null];
    find_signal(name: string): SignalInfo | null;
    find_vfunc(name: string): VFuncInfo | null;
    find_vfunc_using_interfaces(name: string): [VFuncInfo | null, BaseInfo | null];
    get_abstract(): boolean;
    get_class_struct(): StructInfo | null;
    get_constant(n: number): ConstantInfo;
    get_field(n: number): FieldInfo;
    get_final(): boolean;
    get_fundamental(): boolean;
    get_get_value_function_name(): string | null;
    get_interface(n: number): InterfaceInfo;
    get_method(n: number): FunctionInfo;
    get_n_constants(): number;
    get_n_fields(): number;
    get_n_interfaces(): number;
    get_n_methods(): number;
    get_n_properties(): number;
    get_n_signals(): number;
    get_n_vfuncs(): number;
    get_parent(): ObjectInfo | null;
    get_property(n: number): PropertyInfo;
    get_ref_function_name(): string | null;
    get_set_value_function_name(): string | null;
    get_signal(n: number): SignalInfo;
    get_type_init_function_name(): string;
    // Conflicted with GIRepository.RegisteredTypeInfo.get_type_init_function_name
    get_type_init_function_name(...args: never[]): any;
    get_type_name(): string;
    // Conflicted with GIRepository.RegisteredTypeInfo.get_type_name
    get_type_name(...args: never[]): any;
    get_unref_function_name(): string | null;
    get_vfunc(n: number): VFuncInfo;
}
export module PropertyInfo {
    export interface ConstructorProperties extends BaseInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class PropertyInfo extends BaseInfo {
    static $gtype: GObject.GType<PropertyInfo>;

    constructor(properties?: Partial<PropertyInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<PropertyInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    get_flags(): GObject.ParamFlags;
    get_getter(): FunctionInfo | null;
    get_ownership_transfer(): Transfer;
    get_setter(): FunctionInfo | null;
    get_type_info(): TypeInfo;
}
export module RegisteredTypeInfo {
    export interface ConstructorProperties extends BaseInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export abstract class RegisteredTypeInfo extends BaseInfo {
    static $gtype: GObject.GType<RegisteredTypeInfo>;

    constructor(properties?: Partial<RegisteredTypeInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<RegisteredTypeInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    get_g_type(): GObject.GType;
    get_type_init_function_name(): string | null;
    get_type_name(): string | null;
    is_boxed(): boolean;
}
export module Repository {
    export interface ConstructorProperties extends GObject.Object.ConstructorProperties {
        [key: string]: any;
    }
}
export class Repository extends GObject.Object {
    static $gtype: GObject.GType<Repository>;

    constructor(properties?: Partial<Repository.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<Repository.ConstructorProperties>, ...args: any[]): void;

    // Constructors

    static ["new"](): Repository;

    // Members

    enumerate_versions(namespace_: string): string[];
    find_by_error_domain(domain: GLib.Quark): EnumInfo | null;
    find_by_gtype(gtype: GObject.GType): BaseInfo | null;
    find_by_name(namespace_: string, name: string): BaseInfo | null;
    get_c_prefix(namespace_: string): string | null;
    get_dependencies(namespace_: string): string[];
    get_immediate_dependencies(namespace_: string): string[];
    get_info(namespace_: string, idx: number): BaseInfo;
    get_library_path(): string[];
    get_loaded_namespaces(): string[];
    get_n_infos(namespace_: string): number;
    get_object_gtype_interfaces(gtype: GObject.GType): InterfaceInfo[];
    get_search_path(): string[];
    get_shared_libraries(namespace_: string): string[] | null;
    get_typelib_path(namespace_: string): string | null;
    get_version(namespace_: string): string;
    is_registered(namespace_: string, version?: string | null): boolean;
    load_typelib(typelib: Typelib, flags: RepositoryLoadFlags): string;
    prepend_library_path(directory: string): void;
    prepend_search_path(directory: string): void;
    require(namespace_: string, version: string | null, flags: RepositoryLoadFlags): Typelib;
    require_private(
        typelib_dir: string,
        namespace_: string,
        version: string | null,
        flags: RepositoryLoadFlags
    ): Typelib;
    static dump(input_filename: string, output_filename: string): boolean;
    static error_quark(): GLib.Quark;
    static get_option_group(): GLib.OptionGroup;
}
export module SignalInfo {
    export interface ConstructorProperties extends CallableInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class SignalInfo extends CallableInfo {
    static $gtype: GObject.GType<SignalInfo>;

    constructor(properties?: Partial<SignalInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<SignalInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    get_class_closure(): VFuncInfo | null;
    get_flags(): GObject.SignalFlags;
    true_stops_emit(): boolean;
}
export module StructInfo {
    export interface ConstructorProperties extends RegisteredTypeInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class StructInfo extends RegisteredTypeInfo {
    static $gtype: GObject.GType<StructInfo>;

    constructor(properties?: Partial<StructInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<StructInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    find_field(name: string): FieldInfo | null;
    find_method(name: string): FunctionInfo | null;
    get_alignment(): number;
    get_copy_function_name(): string | null;
    get_field(n: number): FieldInfo;
    get_free_function_name(): string | null;
    get_method(n: number): FunctionInfo;
    get_n_fields(): number;
    get_n_methods(): number;
    get_size(): number;
    is_foreign(): boolean;
    is_gtype_struct(): boolean;
}
export module TypeInfo {
    export interface ConstructorProperties extends BaseInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class TypeInfo extends BaseInfo {
    static $gtype: GObject.GType<TypeInfo>;

    constructor(properties?: Partial<TypeInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<TypeInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    argument_from_hash_pointer(hash_pointer: any | null): Argument;
    get_array_fixed_size(): [boolean, number];
    get_array_length_index(): [boolean, number];
    get_array_type(): ArrayType;
    get_interface(): BaseInfo | null;
    get_param_type(n: number): TypeInfo | null;
    get_storage_type(): TypeTag;
    get_tag(): TypeTag;
    hash_pointer_from_argument(arg: Argument): any | null;
    is_pointer(): boolean;
    is_zero_terminated(): boolean;
}
export module UnionInfo {
    export interface ConstructorProperties extends RegisteredTypeInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class UnionInfo extends RegisteredTypeInfo {
    static $gtype: GObject.GType<UnionInfo>;

    constructor(properties?: Partial<UnionInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<UnionInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    find_method(name: string): FunctionInfo | null;
    get_alignment(): number;
    get_copy_function_name(): string | null;
    get_discriminator(n: number): ConstantInfo | null;
    get_discriminator_offset(): [boolean, number];
    get_discriminator_type(): TypeInfo | null;
    get_field(n: number): FieldInfo;
    get_free_function_name(): string | null;
    get_method(n: number): FunctionInfo;
    get_n_fields(): number;
    get_n_methods(): number;
    get_size(): number;
    is_discriminated(): boolean;
}
export module UnresolvedInfo {
    export interface ConstructorProperties extends BaseInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class UnresolvedInfo extends BaseInfo {
    static $gtype: GObject.GType<UnresolvedInfo>;

    constructor(properties?: Partial<UnresolvedInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<UnresolvedInfo.ConstructorProperties>, ...args: any[]): void;
}
export module VFuncInfo {
    export interface ConstructorProperties extends CallableInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class VFuncInfo extends CallableInfo {
    static $gtype: GObject.GType<VFuncInfo>;

    constructor(properties?: Partial<VFuncInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<VFuncInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    get_address(implementor_gtype: GObject.GType): any | null;
    get_flags(): VFuncInfoFlags;
    get_invoker(): FunctionInfo | null;
    get_offset(): number;
    get_signal(): SignalInfo | null;
}
export module ValueInfo {
    export interface ConstructorProperties extends BaseInfo.ConstructorProperties {
        [key: string]: any;
    }
}
export class ValueInfo extends BaseInfo {
    static $gtype: GObject.GType<ValueInfo>;

    constructor(properties?: Partial<ValueInfo.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<ValueInfo.ConstructorProperties>, ...args: any[]): void;

    // Members

    get_value(): number;
}

export class AttributeIter {
    static $gtype: GObject.GType<AttributeIter>;

    constructor(copy: AttributeIter);
}

export class BaseInfoStack {
    static $gtype: GObject.GType<BaseInfoStack>;

    constructor(copy: BaseInfoStack);
}

export class Typelib {
    static $gtype: GObject.GType<Typelib>;

    constructor(bytes: GLib.Bytes | Uint8Array);
    constructor(copy: Typelib);

    // Constructors
    static new_from_bytes(bytes: GLib.Bytes | Uint8Array): Typelib;

    // Members
    get_namespace(): string;
    ref(): Typelib;
    symbol(symbol_name: string): [boolean, any];
    unref(): void;
}

export class Argument {
    static $gtype: GObject.GType<Argument>;

    constructor(
        properties?: Partial<{
            v_boolean?: boolean;
            v_int8?: number;
            v_uint8?: number;
            v_int16?: number;
            v_uint16?: number;
            v_int32?: number;
            v_uint32?: number;
            v_int64?: number;
            v_uint64?: number;
            v_float?: number;
            v_double?: number;
            v_short?: number;
            v_ushort?: number;
            v_int?: number;
            v_uint?: number;
            v_long?: number;
            v_ulong?: number;
            v_ssize?: number;
            v_size?: number;
            v_string?: string;
            v_pointer?: any;
        }>
    );
    constructor(copy: Argument);

    // Fields
    v_boolean: boolean;
    v_int8: number;
    v_uint8: number;
    v_int16: number;
    v_uint16: number;
    v_int32: number;
    v_uint32: number;
    v_int64: number;
    v_uint64: number;
    v_float: number;
    v_double: number;
    v_short: number;
    v_ushort: number;
    v_int: number;
    v_uint: number;
    v_long: number;
    v_ulong: number;
    v_ssize: number;
    v_size: number;
    v_string: string;
    v_pointer: any;
}
