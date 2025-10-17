/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, January 4, 2023.
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

// Â© 2016 and later: Unicode, Inc. and others.
// License & terms of use: http://www.unicode.org/copyright.html
/*
******************************************************************************
* Copyright (C) 2015, International Business Machines
* Corporation and others.  All Rights Reserved.
******************************************************************************
* sharedobject.cpp
*/
#include "sharedobject.h"
#include "mutex.h"
#include "uassert.h"
#include "umutex.h"
#include "unifiedcache.h"

U_NAMESPACE_BEGIN

SharedObject::~SharedObject() {}

UnifiedCacheBase::~UnifiedCacheBase() {}

void
SharedObject::addRef() const {
    umtx_atomic_inc(&hardRefCount);
}

// removeRef Decrement the reference count and delete if it is zero.
//           Note that SharedObjects with a non-null cachePtr are owned by the
//           unified cache, and the cache will be responsible for the actual deletion.
//           The deletion could be as soon as immediately following the
//           update to the reference count, if another thread is running
//           a cache eviction cycle concurrently.
//           NO ACCESS TO *this PERMITTED AFTER REFERENCE COUNT == 0 for cached objects.
//           THE OBJECT MAY ALREADY BE GONE.
void
SharedObject::removeRef() const {
    const UnifiedCacheBase *cache = this->cachePtr;
    int32_t updatedRefCount = umtx_atomic_dec(&hardRefCount);
    U_ASSERT(updatedRefCount >= 0);
    if (updatedRefCount == 0) {
        if (cache) {
            cache->handleUnreferencedObject();
        } else {
            delete this;
        }
    }
}


int32_t
SharedObject::getRefCount() const {
    return umtx_loadAcquire(hardRefCount);
}

void
SharedObject::deleteIfZeroRefCount() const {
    if (this->cachePtr == nullptr && getRefCount() == 0) {
        delete this;
    }
}

U_NAMESPACE_END
