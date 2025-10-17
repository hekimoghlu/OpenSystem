/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, February 8, 2022.
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
 *	objc-references.h
 */

#ifndef _OBJC_REFERENCES_H_
#define _OBJC_REFERENCES_H_

#include "objc-api.h"
#include "objc-config.h"

__BEGIN_DECLS

extern void _objc_associations_init();
extern void _object_set_associative_reference(id object, const void *key, id value, uintptr_t policy);
extern id _object_get_associative_reference(id object, const void *key);
extern void _object_remove_associations(id object, bool deallocating);

__END_DECLS

#endif
