/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 29, 2022.
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
#include "objc-private.h"
#include "DenseMapExtras.h"

static objc::ExplicitInitDenseSet<const char *> namedSelectors;


/***********************************************************************
* sel_init
* Initialize selector tables and register selectors used internally.
**********************************************************************/
void sel_init(size_t selrefCount)
{
#if SUPPORT_PREOPT
    if (PrintPreopt) {
        _objc_inform("PREOPTIMIZATION: using dyld selector opt");
    }
#endif

  namedSelectors.init((unsigned)selrefCount);

    // Register selectors used by libobjc

    mutex_locker_t lock(selLock);

    SEL_cxx_construct = sel_registerNameNoLock(".cxx_construct", NO);
    SEL_cxx_destruct = sel_registerNameNoLock(".cxx_destruct", NO);
}


static SEL sel_alloc(const char *name, bool copy)
{
    lockdebug::assert_locked(&selLock.get());
    return (SEL)(copy ? strdupIfMutable(name) : name);
}


const char *sel_getName(SEL sel) 
{
    if (!sel) return "<null selector>";
    return (const char *)(const void*)sel;
}


unsigned long sel_hash(SEL sel)
{
    unsigned long selAddr = (unsigned long)sel;
#if CONFIG_USE_PREOPT_CACHES
    selAddr ^= (selAddr >> 7);
#endif
    return selAddr;
}


BOOL sel_isMapped(SEL sel) 
{
    if (!sel) return NO;

    const char *name = (const char *)(void *)sel;

    if (sel == _sel_searchBuiltins(name)) return YES;

    mutex_locker_t lock(selLock);
    auto it = namedSelectors.get().find(name);
    return it != namedSelectors.get().end() && (SEL)*it == sel;
}


SEL _sel_searchBuiltins(const char *name)
{
#if SUPPORT_PREOPT
  if (SEL result = (SEL)_dyld_get_objc_selector(name))
    return result;
#endif
    return nil;
}


static SEL __sel_registerName(const char *name, bool shouldLock, bool copy) 
{
    SEL result = 0;

    if (shouldLock) lockdebug::assert_unlocked(&selLock.get());
    else            lockdebug::assert_locked(&selLock.get());

    if (!name) return (SEL)0;

    result = _sel_searchBuiltins(name);
    if (result) return result;
    
    conditional_mutex_locker_t lock(selLock, shouldLock);
	auto it = namedSelectors.get().insert(name);
	if (it.second) {
		// No match. Insert.
		*it.first = (const char *)sel_alloc(name, copy);
	}
	return (SEL)*it.first;
}


SEL sel_registerName(const char *name) {
    return __sel_registerName(name, 1, 1);     // YES lock, YES copy
}

SEL sel_registerNameNoLock(const char *name, bool copy) {
    return __sel_registerName(name, 0, copy);  // NO lock, maybe copy
}


// 2001/1/24
// the majority of uses of this function (which used to return NULL if not found)
// did not check for NULL, so, in fact, never return NULL
//
SEL sel_getUid(const char *name) {
    return __sel_registerName(name, 2, 1);  // YES lock, YES copy
}

SEL sel_lookUpByName(const char *name) {
    if (!name) return (SEL)0;

    SEL result = _sel_searchBuiltins(name);
    if (result) return result;

    mutex_locker_t lock(selLock);
    auto it = namedSelectors.get().find(name);
    if (it == namedSelectors.get().end()) {
        return (SEL)0;
    }
    return (SEL)*it;
}

BOOL sel_isEqual(SEL lhs, SEL rhs)
{
    return bool(lhs == rhs);
}
