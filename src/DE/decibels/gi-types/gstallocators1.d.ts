/**
 * GstAllocators 1.0
 *
 * Generated from 1.0
 */

import * as Gst from "gst1";
import * as GObject from "gobject2";

export const ALLOCATOR_DMABUF: string;
export const ALLOCATOR_FD: string;
export const ALLOCATOR_SHM: string;
export const CAPS_FEATURE_MEMORY_DMABUF: string;
export function dmabuf_memory_get_fd(mem: Gst.Memory): number;
export function drm_dumb_memory_export_dmabuf(mem: Gst.Memory): Gst.Memory;
export function drm_dumb_memory_get_handle(mem: Gst.Memory): number;
export function fd_memory_get_fd(mem: Gst.Memory): number;
export function is_dmabuf_memory(mem: Gst.Memory): boolean;
export function is_drm_dumb_memory(mem: Gst.Memory): boolean;
export function is_fd_memory(mem: Gst.Memory): boolean;
export function is_phys_memory(mem: Gst.Memory): boolean;
export function phys_memory_get_phys_addr(mem: Gst.Memory): never;

export namespace FdMemoryFlags {
    export const $gtype: GObject.GType<FdMemoryFlags>;
}

export enum FdMemoryFlags {
    NONE = 0,
    KEEP_MAPPED = 1,
    MAP_PRIVATE = 2,
    DONT_CLOSE = 4,
}
export module DRMDumbAllocator {
    export interface ConstructorProperties extends Gst.Allocator.ConstructorProperties {
        [key: string]: any;
        drm_device_path: string;
        drmDevicePath: string;
        drm_fd: number;
        drmFd: number;
    }
}
export class DRMDumbAllocator extends Gst.Allocator {
    static $gtype: GObject.GType<DRMDumbAllocator>;

    constructor(properties?: Partial<DRMDumbAllocator.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<DRMDumbAllocator.ConstructorProperties>, ...args: any[]): void;

    // Properties
    get drm_device_path(): string;
    get drmDevicePath(): string;
    get drm_fd(): number;
    get drmFd(): number;

    // Constructors

    static new_with_device_path(drm_device_path: string): DRMDumbAllocator;
    static new_with_fd(drm_fd: number): DRMDumbAllocator;

    // Members

    alloc(drm_fourcc: number, width: number, height: number): [Gst.Memory, number];
    // Conflicted with Gst.Allocator.alloc
    alloc(...args: never[]): any;
    has_prime_export(): boolean;
}
export module DmaBufAllocator {
    export interface ConstructorProperties extends FdAllocator.ConstructorProperties {
        [key: string]: any;
    }
}
export class DmaBufAllocator extends FdAllocator {
    static $gtype: GObject.GType<DmaBufAllocator>;

    constructor(properties?: Partial<DmaBufAllocator.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<DmaBufAllocator.ConstructorProperties>, ...args: any[]): void;

    // Constructors

    static ["new"](): DmaBufAllocator;

    // Members

    static alloc(allocator: Gst.Allocator, fd: number, size: number): Gst.Memory | null;
    static alloc_with_flags(
        allocator: Gst.Allocator,
        fd: number,
        size: number,
        flags: FdMemoryFlags
    ): Gst.Memory | null;
}
export module FdAllocator {
    export interface ConstructorProperties extends Gst.Allocator.ConstructorProperties {
        [key: string]: any;
    }
}
export class FdAllocator extends Gst.Allocator {
    static $gtype: GObject.GType<FdAllocator>;

    constructor(properties?: Partial<FdAllocator.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<FdAllocator.ConstructorProperties>, ...args: any[]): void;

    // Constructors

    static ["new"](): FdAllocator;

    // Members

    static alloc(allocator: Gst.Allocator, fd: number, size: number, flags: FdMemoryFlags): Gst.Memory | null;
}
export module ShmAllocator {
    export interface ConstructorProperties extends FdAllocator.ConstructorProperties {
        [key: string]: any;
    }
}
export class ShmAllocator extends FdAllocator {
    static $gtype: GObject.GType<ShmAllocator>;

    constructor(properties?: Partial<ShmAllocator.ConstructorProperties>, ...args: any[]);
    _init(properties?: Partial<ShmAllocator.ConstructorProperties>, ...args: any[]): void;

    // Members

    static get(): Gst.Allocator | null;
    static init_once(): void;
}

export interface PhysMemoryAllocatorNamespace {
    $gtype: GObject.GType<PhysMemoryAllocator>;
    prototype: PhysMemoryAllocatorPrototype;
}
export type PhysMemoryAllocator = PhysMemoryAllocatorPrototype;
export interface PhysMemoryAllocatorPrototype extends Gst.Allocator {
    // Members

    vfunc_get_phys_addr(mem: Gst.Memory): never;
}

export const PhysMemoryAllocator: PhysMemoryAllocatorNamespace;
