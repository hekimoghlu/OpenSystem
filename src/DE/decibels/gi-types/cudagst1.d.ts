/**
 * CudaGst 1.0
 *
 * Generated from 1.0
 */

import * as GObject from "gobject2";

export const IPC_HANDLE_SIZE: number;
export const TRSF_READ_AS_INTEGER: number;
export const VERSION: number;

export namespace GLDeviceList {
    export const $gtype: GObject.GType<GLDeviceList>;
}

export enum GLDeviceList {
    GL_DEVICE_LIST_ALL = 1,
}

export namespace address_mode {
    export const $gtype: GObject.GType<address_mode>;
}

export enum address_mode {
    WRAP = 0,
    CLAMP = 1,
    MIRROR = 2,
    BORDER = 3,
}

export namespace array_format {
    export const $gtype: GObject.GType<array_format>;
}

export enum array_format {
    INT8 = 1,
    INT16 = 2,
}

export namespace device_attribute {
    export const $gtype: GObject.GType<device_attribute>;
}

export enum device_attribute {
    TEXTURE_ALIGNMENT = 14,
    UNIFIED_ADDRESSING = 41,
    COMPUTE_CAPABILITY_MAJOR = 75,
    COMPUTE_CAPABILITY_MINOR = 76,
    VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED = 102,
    HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
}

export namespace event_flags {
    export const $gtype: GObject.GType<event_flags>;
}

export enum event_flags {
    DEFAULT = 0,
    BLOCKING_SYNC = 1,
    DISABLE_TIMING = 2,
    INTERPROCESS = 4,
}

export namespace filter_mode {
    export const $gtype: GObject.GType<filter_mode>;
}

export enum filter_mode {
    POINT = 0,
    LINEAR = 1,
}

export namespace graphicsMapResourceFlags {
    export const $gtype: GObject.GType<graphicsMapResourceFlags>;
}

export enum graphicsMapResourceFlags {
    NONE = 0,
    READ_ONLY = 1,
    WRITE_DISCARD = 2,
}

export namespace graphicsRegisterFlags {
    export const $gtype: GObject.GType<graphicsRegisterFlags>;
}

export enum graphicsRegisterFlags {
    NONE = 0,
    READ_ONLY = 1,
    WRITE_DISCARD = 2,
    SURFACE_LOAD_STORE = 4,
    TEXTURE_GATHER = 8,
}

export namespace ipcMem_flags {
    export const $gtype: GObject.GType<ipcMem_flags>;
}

export enum ipcMem_flags {
    IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1,
}

export namespace memAccess_flags {
    export const $gtype: GObject.GType<memAccess_flags>;
}

export enum memAccess_flags {
    NONE = 0,
    READ = 1,
    READWRITE = 3,
    MAX = 2147483647,
}

export namespace memAllocationGranularity_flags {
    export const $gtype: GObject.GType<memAllocationGranularity_flags>;
}

export enum memAllocationGranularity_flags {
    MINIMUM = 0,
    RECOMMENDED = 1,
}

export namespace memAllocationHandleType {
    export const $gtype: GObject.GType<memAllocationHandleType>;
}

export enum memAllocationHandleType {
    NONE = 0,
    POSIX_FILE_DESCRIPTOR = 1,
    WIN32 = 2,
    WIN32_KMT = 4,
    MAX = 2147483647,
}

export namespace memAllocationType {
    export const $gtype: GObject.GType<memAllocationType>;
}

export enum memAllocationType {
    INVALID = 0,
    PINNED = 1,
    MAX = 2147483647,
}

export namespace memLocationType {
    export const $gtype: GObject.GType<memLocationType>;
}

export enum memLocationType {
    INVALID = 0,
    DEVICE = 1,
    MAX = 2147483647,
}

export namespace memorytype {
    export const $gtype: GObject.GType<memorytype>;
}

export enum memorytype {
    HOST = 1,
    DEVICE = 2,
    ARRAY = 3,
    UNIFIED = 4,
}

export namespace resourceViewFormat {
    export const $gtype: GObject.GType<resourceViewFormat>;
}

export enum resourceViewFormat {
    RES_VIEW_FORMAT_NONE = 0,
}

export namespace resourcetype {
    export const $gtype: GObject.GType<resourcetype>;
}

export enum resourcetype {
    ARRAY = 0,
    MIPMAPPED_ARRAY = 1,
    LINEAR = 2,
    PITCH2D = 3,
}

export namespace result {
    export const $gtype: GObject.GType<result>;
}

export enum result {
    SUCCESS = 0,
    ERROR_NO_DEVICE = 100,
    ERROR_ALREADY_MAPPED = 208,
    ERROR_NOT_SUPPORTED = 801,
}

export namespace stream_flags {
    export const $gtype: GObject.GType<stream_flags>;
}

export enum stream_flags {
    DEFAULT = 0,
    NON_BLOCKING = 1,
}

export class DA_MEMCPY2D {
    static $gtype: GObject.GType<DA_MEMCPY2D>;

    constructor(copy: DA_MEMCPY2D);

    // Fields
    srcXInBytes: number;
    srcY: number;
    srcMemoryType: memorytype;
    srcHost: any;
    srcDevice: deviceptr;
    srcArray: array;
    srcPitch: number;
    dstXInBytes: number;
    dstY: number;
    dstMemoryType: memorytype;
    dstHost: any;
    dstDevice: deviceptr;
    dstArray: array;
    dstPitch: number;
    WidthInBytes: number;
    Height: number;
}

export class DA_RESOURCE_DESC {
    static $gtype: GObject.GType<DA_RESOURCE_DESC>;

    constructor(copy: DA_RESOURCE_DESC);

    // Fields
    resType: resourcetype;
    flags: number;
}

export class DA_RESOURCE_VIEW_DESC {
    static $gtype: GObject.GType<DA_RESOURCE_VIEW_DESC>;

    constructor(copy: DA_RESOURCE_VIEW_DESC);

    // Fields
    format: resourceViewFormat;
    width: number;
    height: number;
    depth: number;
    firstMipmapLevel: number;
    lastMipmapLevel: number;
    firstLayer: number;
    lastLayer: number;
    reserved: number[];
}

export class DA_TEXTURE_DESC {
    static $gtype: GObject.GType<DA_TEXTURE_DESC>;

    constructor(copy: DA_TEXTURE_DESC);

    // Fields
    addressMode: address_mode[];
    filterMode: filter_mode;
    flags: number;
    maxAnisotropy: number;
    mipmapFilterMode: filter_mode;
    mipmapLevelBias: number;
    minMipmapLevelClamp: number;
    maxMipmapLevelClamp: number;
    borderColor: number[];
    reserved: number[];
}

export class ipcEventHandle {
    static $gtype: GObject.GType<ipcEventHandle>;

    constructor(copy: ipcEventHandle);

    // Fields
    reserved: number[];
}

export class ipcMemHandle {
    static $gtype: GObject.GType<ipcMemHandle>;

    constructor(copy: ipcMemHandle);

    // Fields
    reserved: number[];
}

export class memAccessDesc {
    static $gtype: GObject.GType<memAccessDesc>;

    constructor(copy: memAccessDesc);

    // Fields
    location: memLocation;
    flags: memAccess_flags;
}

export class memAllocationProp {
    static $gtype: GObject.GType<memAllocationProp>;

    constructor(copy: memAllocationProp);

    // Fields
    type: memAllocationType;
    requestedHandleTypes: memAllocationHandleType;
    location: memLocation;
    win32HandleMetaData: any;
    allocFlags: memAllocationPropAllocFlags;
}

export class memAllocationPropAllocFlags {
    static $gtype: GObject.GType<memAllocationPropAllocFlags>;

    constructor(copy: memAllocationPropAllocFlags);

    // Fields
    compressionType: number;
    gpuDirectRDMACapable: number;
    usage: number;
    reserved: Uint8Array;
}

export class memLocation {
    static $gtype: GObject.GType<memLocation>;

    constructor(copy: memLocation);

    // Fields
    type: memLocationType;
    id: number;
}
export type array = any;
export type context = any;
export type device = number;
export type deviceptr = never;
export type event = any;
export type __function = any;
export type graphicsResource = any;
export type mipmappedArray = any;
export type module = any;
export type stream = any;
export type texObject = number;
