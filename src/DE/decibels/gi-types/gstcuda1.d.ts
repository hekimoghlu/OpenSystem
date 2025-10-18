/**
 * GstCuda 1.0
 *
 * Generated from 1.0
 */

import * as Gst from "gst1";
import * as GstVideo from "gstvideo1";
import * as CudaGst from "cudagst1";
import * as GObject from "gobject2";
import * as GLib from "glib2";

export const CAPS_FEATURE_MEMORY_CUDA_MEMORY: string;
export const CUDA_CONTEXT_TYPE: string;
export const CUDA_MEMORY_TYPE_NAME: string;
export const MAP_CUDA: number;
export function buffer_pool_config_get_cuda_alloc_method(config: Gst.Structure): CudaMemoryAllocMethod;
export function buffer_pool_config_get_cuda_stream(config: Gst.Structure): CudaStream | null;
export function buffer_pool_config_set_cuda_alloc_method(config: Gst.Structure, method: CudaMemoryAllocMethod): void;
export function buffer_pool_config_set_cuda_stream(config: Gst.Structure, stream: CudaStream): void;
export function context_new_cuda_context(cuda_ctx: CudaContext): Gst.Context;
export function cuda_create_user_token(): number;
export function cuda_ensure_element_context(
    element: Gst.Element,
    device_id: number,
    cuda_ctx: CudaContext
): [boolean, CudaContext];
export function cuda_handle_context_query(
    element: Gst.Element,
    query: Gst.Query,
    cuda_ctx?: CudaContext | null
): boolean;
export function cuda_handle_set_context(
    element: Gst.Element,
    context: Gst.Context,
    device_id: number,
    cuda_ctx: CudaContext
): [boolean, CudaContext];
export function cuda_load_library(): boolean;
export function cuda_memory_init_once(): void;
export function cuda_nvrtc_compile(source: string): string;
export function cuda_nvrtc_compile_cubin(source: string, device: number): string;
export function cuda_nvrtc_load_library(): boolean;
export function is_cuda_memory(mem: Gst.Memory): boolean;

export namespace CudaGraphicsResourceType {
    export const $gtype: GObject.GType<CudaGraphicsResourceType>;
}

export enum CudaGraphicsResourceType {
    NONE = 0,
    GL_BUFFER = 1,
    D3D11_RESOURCE = 2,
}

export namespace CudaMemoryAllocMethod {
    export const $gtype: GObject.GType<CudaMemoryAllocMethod>;
}

export enum CudaMemoryAllocMethod {
    UNKNOWN = 0,
    MALLOC = 1,
    MMAP = 2,
}

export namespace CudaQuarkId {
    export const $gtype: GObject.GType<CudaQuarkId>;
}

export enum CudaQuarkId {
    GRAPHICS_RESOURCE = 0,
    MAX = 1,
}

export namespace CudaMemoryTransfer {
    export const $gtype: GObject.GType<CudaMemoryTransfer>;
}

export enum CudaMemoryTransfer {
    DOWNLOAD = 1048576,
    UPLOAD = 2097152,
    SYNC = 4194304,
}
export module CudaAllocator {
    export interface ConstructorProperties extends Gst.Allocator.ConstructorProperties {
        [key: string]: any;
    }
}
export class CudaAllocator extends Gst.Allocator {
    static $gtype: GObject.GType<CudaAllocator>;

    constructor(properties?: Partial<CudaAllocator.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<CudaAllocator.ConstructorProperties>, ...args: any[]): void;

    // Members

    alloc(context: CudaContext, stream: CudaStream | null, info: GstVideo.VideoInfo): Gst.Memory | null;
    // Conflicted with Gst.Allocator.alloc
    alloc(...args: never[]): any;
    alloc_wrapped(
        context: CudaContext,
        stream: CudaStream | null,
        info: GstVideo.VideoInfo,
        dev_ptr: CudaGst.deviceptr,
        notify?: GLib.DestroyNotify | null
    ): Gst.Memory;
    set_active(active: boolean): boolean;
    virtual_alloc(
        context: CudaContext,
        stream: CudaStream,
        info: GstVideo.VideoInfo,
        prop: CudaGst.memAllocationProp,
        granularity_flags: CudaGst.memAllocationGranularity_flags
    ): Gst.Memory | null;
    vfunc_set_active(active: boolean): boolean;
}
export module CudaBufferPool {
    export interface ConstructorProperties extends Gst.BufferPool.ConstructorProperties {
        [key: string]: any;
    }
}
export class CudaBufferPool extends Gst.BufferPool {
    static $gtype: GObject.GType<CudaBufferPool>;

    constructor(properties?: Partial<CudaBufferPool.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<CudaBufferPool.ConstructorProperties>, ...args: any[]): void;

    // Fields
    context: CudaContext;
    priv: CudaBufferPoolPrivate;

    // Constructors

    static ["new"](context: CudaContext): CudaBufferPool;
    // Conflicted with Gst.BufferPool.new
    static ["new"](...args: never[]): any;
}
export module CudaContext {
    export interface ConstructorProperties extends Gst.Object.ConstructorProperties {
        [key: string]: any;
        cuda_device_id: number;
        cudaDeviceId: number;
        os_handle: boolean;
        osHandle: boolean;
        virtual_memory: boolean;
        virtualMemory: boolean;
    }
}
export class CudaContext extends Gst.Object {
    static $gtype: GObject.GType<CudaContext>;

    constructor(properties?: Partial<CudaContext.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<CudaContext.ConstructorProperties>, ...args: any[]): void;

    // Properties
    get cuda_device_id(): number;
    get cudaDeviceId(): number;
    get os_handle(): boolean;
    get osHandle(): boolean;
    get virtual_memory(): boolean;
    get virtualMemory(): boolean;

    // Fields
    object: Gst.Object;

    // Constructors

    static ["new"](device_id: number): CudaContext;
    static new_wrapped(handler: CudaGst.context, device: CudaGst.device): CudaContext;

    // Members

    can_access_peer(peer: CudaContext): boolean;
    get_handle(): any | null;
    get_texture_alignment(): number;
    push(): boolean;
    static pop(cuda_ctx: CudaGst.context): boolean;
}
export module CudaPoolAllocator {
    export interface ConstructorProperties extends CudaAllocator.ConstructorProperties {
        [key: string]: any;
    }
}
export class CudaPoolAllocator extends CudaAllocator {
    static $gtype: GObject.GType<CudaPoolAllocator>;

    constructor(properties?: Partial<CudaPoolAllocator.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<CudaPoolAllocator.ConstructorProperties>, ...args: any[]): void;

    // Fields
    context: CudaContext;
    stream: CudaStream;
    info: GstVideo.VideoInfo;

    // Constructors

    static ["new"](context: CudaContext, stream: CudaStream | null, info: GstVideo.VideoInfo): CudaPoolAllocator;
    static new_for_virtual_memory(
        context: CudaContext,
        stream: CudaStream | null,
        info: GstVideo.VideoInfo,
        prop: CudaGst.memAllocationProp,
        granularity_flags: CudaGst.memAllocationGranularity_flags
    ): CudaPoolAllocator;

    // Members

    acquire_memory(): [Gst.FlowReturn, Gst.Memory];
}

export class CudaAllocatorPrivate {
    static $gtype: GObject.GType<CudaAllocatorPrivate>;

    constructor(copy: CudaAllocatorPrivate);
}

export class CudaBufferPoolPrivate {
    static $gtype: GObject.GType<CudaBufferPoolPrivate>;

    constructor(copy: CudaBufferPoolPrivate);
}

export class CudaContextPrivate {
    static $gtype: GObject.GType<CudaContextPrivate>;

    constructor(copy: CudaContextPrivate);
}

export class CudaGraphicsResource {
    static $gtype: GObject.GType<CudaGraphicsResource>;

    constructor(copy: CudaGraphicsResource);

    // Fields
    cuda_context: CudaContext;
    graphics_context: Gst.Object;
    type: CudaGraphicsResourceType;
    resource: CudaGst.graphicsResource;
    flags: CudaGst.graphicsRegisterFlags;
    registered: boolean;
    mapped: boolean;
}

export class CudaMemory {
    static $gtype: GObject.GType<CudaMemory>;

    constructor(copy: CudaMemory);

    // Fields
    mem: Gst.Memory;
    context: CudaContext;
    info: GstVideo.VideoInfo;

    // Members
    ["export"](): [boolean, any];
    get_alloc_method(): CudaMemoryAllocMethod;
    get_stream(): CudaStream | null;
    get_texture(plane: number, filter_mode: CudaGst.filter_mode): [boolean, CudaGst.texObject];
    get_token_data(token: number): any | null;
    get_user_data(): any | null;
    set_token_data(token: number, data?: any | null): void;
    sync(): void;
    static init_once(): void;
}

export class CudaMemoryPrivate {
    static $gtype: GObject.GType<CudaMemoryPrivate>;

    constructor(copy: CudaMemoryPrivate);
}

export class CudaPoolAllocatorPrivate {
    static $gtype: GObject.GType<CudaPoolAllocatorPrivate>;

    constructor(copy: CudaPoolAllocatorPrivate);
}

export class CudaStream {
    static $gtype: GObject.GType<CudaStream>;

    constructor(context: CudaContext);
    constructor(copy: CudaStream);

    // Fields
    parent: Gst.MiniObject;
    context: CudaContext;

    // Constructors
    static ["new"](context: CudaContext): CudaStream;

    // Members
    get_handle(): CudaGst.stream;
    ref(): CudaStream;
    unref(): void;
}

export class CudaStreamPrivate {
    static $gtype: GObject.GType<CudaStreamPrivate>;

    constructor(copy: CudaStreamPrivate);
}
