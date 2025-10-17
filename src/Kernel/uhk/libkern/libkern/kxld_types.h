/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, October 4, 2023.
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
#ifndef _KXLD_TYPES_H
#define _KXLD_TYPES_H

#include <stdarg.h>
#include <stdint.h>
#include <mach/boolean.h>       // boolean_t
#include <mach/kern_return.h>

/*******************************************************************************
* Macros
*******************************************************************************/

/* For 32-bit-specific linking code */
#if (!KERNEL || !__LP64__)
    #define KXLD_USER_OR_ILP32 1
#endif

/* For 64-bit-specific linking code */
#if (!KERNEL || __LP64__)
    #define KXLD_USER_OR_LP64 1
#endif

/* For i386-specific linking code */
#if (!KERNEL || __i386__)
    #define KXLD_USER_OR_I386 1
#endif

/* For x86_64-specific linking code */
#if (!KERNEL || __x86_64__)
    #define KXLD_USER_OR_X86_64 1
#endif

/* For arm-specific linking code */
#if (!KERNEL || __arm__)
    #define KXLD_USER_OR_ARM 1
#endif

/* For arm64-specific linking code */
#if (!KERNEL || __arm64__)
    #define KXLD_USER_OR_ARM64 1
#endif

/* For linking code specific to architectures that support common symbols */
#if (!KERNEL || __i386__)
    #define KXLD_USER_OR_COMMON 1
#endif

/* For linking code specific to architectures that support strict patching */
    #define KXLD_USER_OR_STRICT_PATCHING 1

/* For linking code specific to architectures that use MH_OBJECT */
#if (!KERNEL || __i386__)
    #define KXLD_USER_OR_OBJECT 1
#endif

/* For linking code specific to architectures that use MH_KEXT_BUNDLE */
#define KXLD_USER_OR_BUNDLE 1

/* We no longer need to generate our own GOT for any architectures, but the code
 * required to do this will be saved inside this macro.
 */
#define KXLD_USER_OR_GOT 0

/* for building the dysymtab command generation into the dylib */
#if (!KERNEL)
    #define KXLD_PIC_KEXTS 1
//    #define SPLIT_KEXTS 1
    #define SPLIT_KEXTS_DEBUG 0
#endif

/*******************************************************************************
* Types
*******************************************************************************/

/* Maintains linker state across links.  One context should be allocated for
 * each link thread.
 */
typedef struct kxld_context KXLDContext;

/* Unless we're in a 32-bit kernel, all internal math is performed in 64 bits
 * and cast to smaller values as needed by the architecture for which we are
 * linking.  All returned arguments should be handled similarly.
 * Note: This size can be increased for future architectural size increases
 */
#if KERNEL && !__LP64__
typedef uint32_t kxld_addr_t;
typedef uint32_t kxld_size_t;
#else
typedef uint64_t kxld_addr_t;
typedef uint64_t kxld_size_t;
#endif /* KERNEL && !__LP64__ */

typedef struct splitKextLinkInfo {
	u_char *        kextExecutable; // kext we will link
	size_t          kextSize;       // size of kextExecutable
	u_char *        linkedKext;     // linked kext
	size_t          linkedKextSize; // size of linkedKext
	uint64_t        vmaddr_TEXT;    // vmaddr of kext __TEXT segment
	uint64_t        vmaddr_TEXT_EXEC;// vmaddr of kext __TEXT_EXEC segment
	uint64_t        vmaddr_DATA;    // vmaddr of kext __DATA segment
	uint64_t        vmaddr_DATA_CONST;// vmaddr of kext __DATA_CONST segment
	uint64_t        vmaddr_LINKEDIT;// vmaddr of kext __LINKEDIT segment
	uint64_t        vmaddr_LLVM_COV;// vmaddr of kext __LLVM_COV segment
	uint32_t        kaslr_offsets_count;// offsets into the kext to slide
	uint32_t *      kaslr_offsets;  // offsets into the kext to slide
} splitKextLinkInfo;

/* Flags for general linker behavior */
enum kxld_flags {
	kKxldFlagDefault = 0x0,
	kKXLDFlagIncludeRelocs = 0x01
};
typedef enum kxld_flags KXLDFlags;

/* Flags for the allocation callback */
enum kxld_allocate_flags {
	kKxldAllocateDefault = 0x0,
	kKxldAllocateWritable = 0x1,    /* kxld may write into the allocated memory */
};
typedef enum kxld_allocate_flags KXLDAllocateFlags;

/* This specifies the function type of the callback that the linker uses to get
 * the base address and allocated memory for relocation and linker output,
 * respectively.  Note that it is compatible with the standard allocators (e.g.
 * malloc).
 */
typedef kxld_addr_t (*KXLDAllocateCallback)(size_t size,
    KXLDAllocateFlags *flags, void *user_data);

/* Flags for the logging callback */
typedef enum kxld_log_subsystem {
	kKxldLogLinking = 0x0,
	kKxldLogPatching = 0x01
} KXLDLogSubsystem;

typedef enum kxld_log_level {
	kKxldLogExplicit = 0x0,
	kKxldLogErr = 0x1,
	kKxldLogWarn = 0x2,
	kKxldLogBasic = 0x3,
	kKxldLogDetail = 0x4,
	kKxldLogDebug = 0x5
} KXLDLogLevel;

/* This structure is used to describe a dependency kext. The kext field
 * is a pointer to the binary executable of the dependency. The interface
 * field is a pointer to an optional interface kext that restricts the
 * symbols that may be accessed in the dependency kext.
 *
 * For example, to use this structure with the KPIs, set the kext field
 * to point to the kernel's Mach-O binary, and set interface to point
 * to the KPI's Mach-O binary.
 */
typedef struct kxld_dependency {
	u_char      * kext;
	u_long        kext_size;
	char        * kext_name;
	u_char      * interface;
	u_long        interface_size;
	char        * interface_name;
	boolean_t     is_direct_dependency;
} KXLDDependency;

typedef void (*KXLDLoggingCallback) (KXLDLogSubsystem sys, KXLDLogLevel level,
    const char *format, va_list ap, void *user_data);

#endif /* _KXLD_TYPES_H */
