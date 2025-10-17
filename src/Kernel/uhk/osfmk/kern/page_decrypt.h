/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, February 22, 2024.
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
#ifdef  KERNEL_PRIVATE

#ifndef _KERN_PAGE_DECRYPT_H
#define _KERN_PAGE_DECRYPT_H

#include <mach/machine.h>

/*
 * Interface for DSMOS
 */
typedef int     (*dsmos_page_transform_hook_t) (const void *, void*, unsigned long long, void *);
extern  void    dsmos_page_transform_hook(dsmos_page_transform_hook_t hook);    /* exported */

extern  int     dsmos_page_transform(const void *, void*, unsigned long long, void*);


/*
 * Interface for text decryption family
 */
struct pager_crypt_info {
	/* Decrypt one page */
	int     (*page_decrypt)(const void *src_vaddr, void *dst_vaddr,
	    unsigned long long src_offset, void *crypt_ops);
	/* Pager using this crypter terminates - crypt module not needed anymore */
	void    (*crypt_end)(void *crypt_ops);
	/* Private data for the crypter */
	void    *crypt_ops;
	volatile int    crypt_refcnt;
};
typedef struct pager_crypt_info pager_crypt_info_t;

typedef enum {
	CRYPT_ORIGIN_ANY,
	CRYPT_ORIGIN_APP_LAUNCH,
	CRYPT_ORIGIN_LIBRARY_LOAD,
	CRYPT_ORIGIN_MAX,
} crypt_origin_t;

struct crypt_file_data {
	char          *filename;
	cpu_type_t     cputype;
	cpu_subtype_t  cpusubtype;
	crypt_origin_t origin;
};
typedef struct crypt_file_data crypt_file_data_t;

typedef int (*text_crypter_create_hook_t)(struct pager_crypt_info *crypt_info,
    const char *id, void *crypt_data);
extern void text_crypter_create_hook_set(text_crypter_create_hook_t hook);
extern text_crypter_create_hook_t text_crypter_create;

#endif  /* _KERN_PAGE_DECRYPT_H */

#endif  /* KERNEL_PRIVATE */
