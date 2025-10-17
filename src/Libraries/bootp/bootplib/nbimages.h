/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 8, 2022.
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
/*
 * nbimages.h
 * - NetBoot image list routines
 */

#ifndef _S_NBIMAGES_H
#define _S_NBIMAGES_H

#include <net/ethernet.h>
#include "nbsp.h"
#include "bsdp.h"
#include <sys/types.h>

typedef enum {
    kNBImageTypeNone = 0,
    kNBImageTypeClassic,
    kNBImageTypeNFS,
    kNBImageTypeHTTP,
    kNBImageTypeBootFileOnly,
} NBImageType;

typedef union {
    struct {
	boolean_t	indirect;
	const char *	root_path;
    } nfs;
    struct {
	boolean_t	indirect;
	const char *	root_path;
	const char *	root_path_esc;
    } http;
    struct {
	const char *	shared;
	const char *	private;
    } classic;
} NBImageTypeInfo;

typedef struct {
    NBSPEntryRef	sharepoint;
    const char *	arch;
    const char * *	archlist;
    int			archlist_count;
    char *		dir_name;	/* relative to sharepoint */
    char *		dir_name_esc; 	/* spaces/etc. escaped e.g. %20 */
    char *		name;
    int			name_length;
    bsdp_image_id_t	image_id;
    const char *	bootfile;
    boolean_t		ppc_bootfile_no_subdir;
    NBImageType		type;
    NBImageTypeInfo	type_info;
    boolean_t		is_default;
    boolean_t		diskless;
    boolean_t		filter_only;
    const char * *	sysids;
    int			sysids_count;
    const struct ether_addr *	enabled_mac_addresses;
    int			enabled_mac_addresses_count;
    const struct ether_addr *	disabled_mac_addresses;
    int			disabled_mac_addresses_count;
    struct in_addr	load_balance_ip;
} NBImageEntry, * NBImageEntryRef;

boolean_t	NBImageEntry_supported_sysid(NBImageEntryRef entry, 
					     const char * arch,
					     const char * sysid,
					     const struct ether_addr * ether_addr);
struct NBImageList_s;
typedef struct NBImageList_s * NBImageListRef;

int		NBImageList_count(NBImageListRef list);
NBImageEntryRef	NBImageList_element(NBImageListRef list, int i);
NBImageEntryRef NBImageList_elementWithID(NBImageListRef list, bsdp_image_id_t);
NBImageListRef	NBImageList_init(NBSPListRef sharepoints,
				 boolean_t allow_diskless);
void		NBImageList_free(NBImageListRef * list);
void		NBImageList_print(NBImageListRef images);
NBImageEntryRef NBImageList_default(NBImageListRef images, 
				    const char * arch, const char * sysid,
				    const struct ether_addr * ether,
				    const u_int16_t * attrs, int n_attrs);
boolean_t	NBImageEntry_attributes_match(NBImageEntryRef entry,
					      const u_int16_t * attrs,
					      int n_attrs);

#endif /* _S_NBIMAGES_H */
