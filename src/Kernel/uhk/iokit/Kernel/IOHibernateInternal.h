/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 12, 2022.
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
#include <stdint.h>
#include <IOKit/IOHibernatePrivate.h>

#ifdef __cplusplus


enum { kIOHibernateAESKeySize = 16 };  /* bytes */

// srcBuffer has to be big enough for a source page, the WKDM
// compressed output, and a scratch page needed by WKDM
#define HIBERNATION_SRC_BUFFER_SIZE (2 * page_size + WKdm_SCRATCH_BUF_SIZE_INTERNAL)

struct IOHibernateVars {
	hibernate_page_list_t *             page_list;
	hibernate_page_list_t *             page_list_wired;
	hibernate_page_list_t *             page_list_pal;
	class IOBufferMemoryDescriptor *    ioBuffer;
	class IOBufferMemoryDescriptor *    srcBuffer;
	class IOBufferMemoryDescriptor *    handoffBuffer;
	class IOMemoryDescriptor *          previewBuffer;
	OSData *                            previewData;
	OSObject *                          saveBootDevice;

	struct IOPolledFileIOVars *         fileVars;
	uint64_t                            fileMinSize;
	uint64_t                            fileMaxSize;
	vm_offset_t                         videoMapping;
	vm_size_t                           videoAllocSize;
	vm_size_t                           videoMapSize;
	uint8_t *                           consoleMapping;
	uint8_t                             haveFastBoot;
	uint8_t                             saveBootAudioVolume;
	uint8_t                             hwEncrypt;
	uint8_t                             wiredCryptKey[kIOHibernateAESKeySize];
	uint8_t                             cryptKey[kIOHibernateAESKeySize];
	size_t                              volumeCryptKeySize;
	uint8_t                             volumeCryptKey[64];
};
typedef struct IOHibernateVars IOHibernateVars;

#endif          /* __cplusplus */

enum{
	kIOHibernateTagSignature = 0x53000000u,
	kIOHibernateTagSigMask   = 0xff000000u,
	kIOHibernateTagLength    = 0x00007fffu,
	kIOHibernateTagSKCrypt   = 0x00800000u,
};

#ifdef __cplusplus
extern "C"
#endif          /* __cplusplus */
uint32_t
hibernate_sum_page(uint8_t *buf, uint32_t ppnum);

#if defined(__i386__) || defined(__x86_64__)
extern vm_offset_t segHIBB;
extern unsigned long segSizeHIB;
#elif defined(__arm64__)
extern vm_offset_t sectHIBTEXTB;
extern unsigned long sectSizeHIBTEXT;
#endif

extern ppnum_t gIOHibernateHandoffPages[];
extern const uint32_t gIOHibernateHandoffPageCount;

// max address that can fit in a ppnum_t
#define IO_MAX_PAGE_ADDR        (((uint64_t) UINT_MAX) << PAGE_SHIFT)
// atop() returning ppnum_t
#define atop_64_ppnum(x) ((ppnum_t)((uint64_t)(x) >> PAGE_SHIFT))
