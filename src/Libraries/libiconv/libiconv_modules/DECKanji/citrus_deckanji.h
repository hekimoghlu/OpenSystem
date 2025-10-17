/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, January 15, 2023.
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
#ifndef _CITRUS_DECKANJI_H_
#define _CITRUS_DECKANJI_H_

#ifdef __APPLE__
#include <sys/cdefs.h>
#include <sys/queue.h>

#include <string.h> /* memcpy */

#include "citrus_namespace.h"
#include "citrus_types.h"
#include "citrus_module.h"
#include "citrus_hash.h"
#include "citrus_region.h"
#include "citrus_stdenc.h"

extern struct _citrus_stdenc_ops _citrus_DECKanji_stdenc_ops;
#endif

__BEGIN_DECLS
_CITRUS_STDENC_GETOPS_FUNC(DECKanji);
__END_DECLS

#endif
