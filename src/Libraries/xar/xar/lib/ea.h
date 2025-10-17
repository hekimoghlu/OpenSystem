/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 22, 2023.
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

#ifndef _XAR_EA_H_
#define _XAR_EA_H_

#include "xar.h"
#include "filetree.h"

typedef struct __xar_ea_t *xar_ea_t;

xar_ea_t xar_ea_new(xar_file_t f, const char *name);
int32_t xar_ea_pset(xar_file_t f, xar_ea_t e, const char *key, const char *value);
int32_t xar_ea_pget(xar_ea_t e, const char *key, const char **value);
xar_prop_t xar_ea_root(xar_ea_t e);
xar_prop_t xar_ea_find(xar_file_t f, const char *name);

#endif /* _XAR_EA_H_ */
