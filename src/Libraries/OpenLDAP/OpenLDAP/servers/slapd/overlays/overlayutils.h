/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, July 13, 2022.
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

#ifndef __OVERLAYUTILS_H__
#define __OVERLAYUTILS_H__

#include "portable.h"
#include <stdio.h>

#include <ac/string.h>
#include <ac/ctype.h>
#include "slap.h"
#include "ldif.h"
#include "config.h"

void dump_berval( struct berval *bv );
void dump_berval_array(BerVarray bva);
void dump_slap_attr_desc(AttributeDescription *desc);
void dump_slap_attr(Attribute *attr);
void dump_slap_entry(Entry *ent);
void dump_req_bind_s(req_bind_s *req);
void dump_req_add_s(req_add_s *req);

#endif /* __OVERLAYUTILS_H__ */
