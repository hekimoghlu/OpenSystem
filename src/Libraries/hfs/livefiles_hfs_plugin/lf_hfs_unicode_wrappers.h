/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, January 3, 2023.
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

//
//  lf_hfs_unicode_wrappers.h
//  livefiles_hfs
//
//  Created by Yakov Ben Zaken on 22/03/2018.
//

#ifndef lf_hfs_unicode_wrappers_h
#define lf_hfs_unicode_wrappers_h

#include <stdio.h>
#include "lf_hfs_defs.h"
#include "lf_hfs_file_mgr_internal.h"

int32_t FastUnicodeCompare      ( register ConstUniCharArrayPtr str1, register ItemCount len1, register ConstUniCharArrayPtr str2, register ItemCount len2);

int32_t UnicodeBinaryCompare    ( register ConstUniCharArrayPtr str1, register ItemCount len1, register ConstUniCharArrayPtr str2, register ItemCount len2 );

HFSCatalogNodeID GetEmbeddedFileID( ConstStr31Param filename, u_int32_t length, u_int32_t *prefixLength );

OSErr ConvertUnicodeToUTF8Mangled(ByteCount srcLen, ConstUniCharArrayPtr srcStr, ByteCount maxDstLen,
                                  ByteCount *actualDstLen, unsigned char* dstStr, HFSCatalogNodeID cnid);

u_int32_t
CountFilenameExtensionChars( const unsigned char * filename, u_int32_t length );


#endif /* lf_hfs_unicode_wrappers_h */
