/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 1, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef __IOSKYWALKSUPPORT_H
#define __IOSKYWALKSUPPORT_H

#ifdef KERNEL_PRIVATE
#include <sys/cdefs.h>
#include <IOKit/IOReturn.h>

#ifdef __cplusplus
class IOMemoryDescriptor;
typedef IOMemoryDescriptor *        IOSKMemoryRef;
class IOSKArena;
typedef IOSKArena *                 IOSKArenaRef;
class IOSKRegion;
typedef IOSKRegion *                IOSKRegionRef;
class IOSKMapper;
typedef IOSKMapper *                IOSKMapperRef;
#define IOSK_CONSUMED LIBKERN_CONSUMED
#else /* !__cplusplus */
typedef struct IOMemoryDescriptor * IOSKMemoryRef;
typedef struct IOSKArena *          IOSKArenaRef;
typedef struct IOSKRegion *         IOSKRegionRef;
typedef struct IOSKMapper *         IOSKMapperRef;
#define IOSK_CONSUMED
#endif /* !__cplusplus */

typedef IOSKMemoryRef   IOSKMemoryDescriptor;

#if defined(__x86_64__) && defined(__cplusplus)
const OSSymbol *  IOSKCopyKextIdentifierWithAddress( vm_address_t address );
#endif

#ifdef XNU_KERNEL_PRIVATE
#ifdef __cplusplus
class IOMemoryMap;
typedef IOMemoryMap *               IOSKMemoryMapRef;
#else /* !__cplusplus */
typedef struct IOMemoryMap *        IOSKMemoryMapRef;
#endif /* !__cplusplus */

typedef IOSKMemoryRef   IOSKMemoryArrayRef;
typedef IOSKMemoryRef   IOSKMemoryBufferRef;
typedef uint32_t        IOSKSize;
typedef uint32_t        IOSKIndex;
typedef uint32_t        IOSKCount;
typedef uint32_t        IOSKOffset;

typedef struct {
	boolean_t user_writable;        /* writable by user task */
	boolean_t kernel_writable;      /* writable by kernel task */
	boolean_t iodir_in;             /* direction: device-to-host */
	boolean_t iodir_out;            /* direction: host-to-device */
	boolean_t purgeable;            /* purgeable (not wired) */
	boolean_t inhibitCache;         /* cache-inhibit */
	boolean_t physcontig;           /* physically contiguous */
	boolean_t puredata;             /* data only, no pointers */
	boolean_t threadSafe;           /* thread safe */
	boolean_t memtag;               /* memtag enabled */
} IOSKMemoryBufferSpec;

typedef struct {
	boolean_t noRedirect;
} IOSKRegionSpec;

__BEGIN_DECLS

IOSKMemoryBufferRef IOSKMemoryBufferCreate( mach_vm_size_t capacity,
    const IOSKMemoryBufferSpec * spec,
    mach_vm_address_t * kvaddr );

IOSKMemoryArrayRef  IOSKMemoryArrayCreate( const IOSKMemoryRef refs[__counted_by(count)],
    uint32_t count );

void                IOSKMemoryDestroy( IOSK_CONSUMED IOSKMemoryRef reference );

IOSKMemoryMapRef    IOSKMemoryMapToTask( IOSKMemoryRef reference,
    task_t intoTask,
    mach_vm_address_t * mapAddr,
    mach_vm_size_t * mapSize );

IOSKMemoryMapRef    IOSKMemoryMapToKernelTask( IOSKMemoryRef reference,
    mach_vm_address_t * mapAddr,
    mach_vm_size_t * mapSize );

void                IOSKMemoryMapDestroy(
	IOSK_CONSUMED IOSKMemoryMapRef reference );

IOReturn            IOSKMemoryReclaim( IOSKMemoryRef reference );
IOReturn            IOSKMemoryDiscard( IOSKMemoryRef reference );

IOReturn            IOSKMemoryWire( IOSKMemoryRef reference );
IOReturn            IOSKMemoryUnwire( IOSKMemoryRef reference );

IOSKArenaRef
IOSKArenaCreate( IOSKRegionRef * regionList, IOSKCount regionCount );

void
IOSKArenaDestroy( IOSK_CONSUMED IOSKArenaRef arena );

void
IOSKArenaRedirect( IOSKArenaRef arena );

IOSKRegionRef
IOSKRegionCreate( const IOSKRegionSpec * regionSpec,
    IOSKSize segmentSize, IOSKCount segmentCount );

void
IOSKRegionDestroy( IOSK_CONSUMED IOSKRegionRef region );

IOReturn
IOSKRegionSetBuffer( IOSKRegionRef region, IOSKIndex segmentIndex,
    IOSKMemoryBufferRef buffer );

void
IOSKRegionClearBuffer( IOSKRegionRef region, IOSKIndex segmentIndex );

void
IOSKRegionClearBufferDebug( IOSKRegionRef region, IOSKIndex segmentIndex,
    IOSKMemoryBufferRef * prevBufferRef );

IOSKMapperRef
IOSKMapperCreate( IOSKArenaRef arena, task_t task );

void
IOSKMapperDestroy( IOSK_CONSUMED IOSKMapperRef mapper );

void
IOSKMapperRedirect( IOSKMapperRef mapper );

IOReturn
IOSKMapperGetAddress( IOSKMapperRef mapper,
    mach_vm_address_t * address, mach_vm_size_t * size );

boolean_t
IOSKBufferIsWired( IOSKMemoryBufferRef buffer );

__END_DECLS
#endif /* XNU_KERNEL_PRIVATE */
#endif /* KERNEL_PRIVATE */
#endif /* __IOSKYWALKSUPPORT_H */
