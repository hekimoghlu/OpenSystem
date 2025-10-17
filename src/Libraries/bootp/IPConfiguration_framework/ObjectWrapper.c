/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 3, 2024.
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
#include <stdatomic.h>
#include "symbol_scope.h"
#include "cfutil.h"
#include "IPConfigurationLog.h"
#include "ObjectWrapper.h"

struct ObjectWrapper {
    const void *	obj;
    int32_t		retain_count;
};

PRIVATE_EXTERN const void *
ObjectWrapperRetain(const void * info)
{
#ifdef DEBUG
    int32_t		new_val;
#endif
    ObjectWrapperRef 	wrapper = (ObjectWrapperRef)info;

#ifdef DEBUG
    new_val =
#endif
    atomic_fetch_add_explicit((_Atomic int32_t *)&wrapper->retain_count, 1,
			      memory_order_relaxed);
#ifdef DEBUG
    /* Apparently, atomic_fetch_add_explicit() returns the old value,
     * not the new one. A +1 is needed here.
     */
    printf("wrapper retain (%d)\n", (new_val + 1));
#endif
    return (info);
}

PRIVATE_EXTERN const void *
ObjectWrapperGetObject(ObjectWrapperRef wrapper)
{
    return (wrapper->obj);
}

PRIVATE_EXTERN void
ObjectWrapperClearObject(ObjectWrapperRef wrapper)
{
    wrapper->obj = NULL;
}

PRIVATE_EXTERN ObjectWrapperRef
ObjectWrapperAlloc(const void * obj)
{
    ObjectWrapperRef	wrapper;

    wrapper = (ObjectWrapperRef)malloc(sizeof(*wrapper));
    wrapper->obj = obj;
    wrapper->retain_count = 1;
    return (wrapper);
}

PRIVATE_EXTERN void
ObjectWrapperRelease(const void * info)
{
    int32_t		new_val;
    ObjectWrapperRef 	wrapper = (ObjectWrapperRef)info;

    new_val = atomic_fetch_sub_explicit((_Atomic int32_t *)&wrapper->retain_count, 1,
					memory_order_relaxed) - 1;
#ifdef DEBUG
    printf("wrapper release (%d)\n", new_val);
#endif
    if (new_val == 0) {
#ifdef DEBUG
	printf("wrapper free\n");
#endif
	free(wrapper);
    }
    else if (new_val < 0) {
	IPConfigLogFL(LOG_NOTICE,
		      "IPConfigurationService: retain count already zero");
	abort();
    }
    return;
}
