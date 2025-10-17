/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 17, 2023.
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
#ifndef _SECAKSWRAPPERS_H_
#define _SECAKSWRAPPERS_H_

#include <TargetConditionals.h>
#include "utilities/SecCFError.h"
#include <AssertMacros.h>
#include <dispatch/dispatch.h>

#include <CoreFoundation/CFData.h>

#if defined(USE_KEYSTORE)
#define TARGET_HAS_KEYSTORE USE_KEYSTORE

#else

#if RC_HORIZON
  #define TARGET_HAS_KEYSTORE 0
#elif TARGET_OS_SIMULATOR
  #define TARGET_HAS_KEYSTORE 0
#elif TARGET_OS_OSX
  #if TARGET_CPU_X86
    #define TARGET_HAS_KEYSTORE 0
  #else
    #define TARGET_HAS_KEYSTORE 1
  #endif
#elif TARGET_OS_IPHONE
  #define TARGET_HAS_KEYSTORE 1
#else
  #error "unknown keystore status for this platform"
#endif

#endif // USE_KEYSTORE

#if __has_include(<AppleKeyStore/libaks.h>)
#include <AppleKeyStore/libaks.h>
#else
#undef INCLUDE_MOCK_AKS
#define INCLUDE_MOCK_AKS 1
#endif

#if __has_include(<MobileKeyBag/MobileKeyBag.h>)
#include <MobileKeyBag/MobileKeyBag.h>
#else
#undef INCLUDE_MOCK_AKS
#define INCLUDE_MOCK_AKS 1
#endif

#if INCLUDE_MOCK_AKS
#include "tests/secdmockaks/mockaks.h"
#endif


bool hwaes_key_available(void);

//
// MARK: User lock state
//

enum {
    // WARNING: Do not use this from the system session. It will likely not do the right thing.
    // Current uses are from SOS, CKP, CJR & LKA, none of which are used in the system session.
    // For LKA, see also comment in SecItemServer.c
#if TARGET_OS_OSX && TARGET_HAS_KEYSTORE
    user_only_keybag_handle = session_keybag_handle,
#else // either embedded os with keystore, or simulator
    user_only_keybag_handle = device_keybag_handle,
#endif
};

extern const char * const kUserKeybagStateChangeNotification;

static inline bool SecAKSGetLockedState(keybag_handle_t handle, keybag_state_t *state, CFErrorRef* error)
{
    kern_return_t status = aks_get_lock_state(handle, state);

    return SecKernError(status, error, CFSTR("aks_get_lock_state failed: %x"), status);
}

// returns true if any of the bits in bits is set in the current state of the user bag
static inline bool SecAKSLockedAnyStateBitIsSet(keybag_handle_t handle, bool* isSet, keybag_state_t bits, CFErrorRef* error)
{
    keybag_state_t state;
    bool success = SecAKSGetLockedState(handle, &state, error);
    
    require_quiet(success, exit);
    
    if (isSet)
        *isSet = (state & bits);
    
exit:
    return success;

}

static inline bool SecAKSGetIsLocked(keybag_handle_t handle, bool* isLocked, CFErrorRef* error)
{
    return SecAKSLockedAnyStateBitIsSet(handle, isLocked, keybag_state_locked, error);
}

static inline bool SecAKSGetIsUnlocked(keybag_handle_t handle, bool* isUnlocked, CFErrorRef* error)
{
    bool isLocked = false;
    bool success = SecAKSGetIsLocked(handle, &isLocked, error);

    if (success && isUnlocked)
        *isUnlocked = !isLocked;

    return success;
}

static inline bool SecAKSGetHasBeenUnlocked(keybag_handle_t handle, bool* hasBeenUnlocked, CFErrorRef* error)
{
    return SecAKSLockedAnyStateBitIsSet(handle, hasBeenUnlocked, keybag_state_been_unlocked, error);
}

bool SecAKSDoWithKeybagLockAssertion(keybag_handle_t handle, CFErrorRef *error, dispatch_block_t action);

//just like SecAKSDoWithKeybagLockAssertion, but always perform action regardless if we got the assertion or not
bool SecAKSDoWithKeybagLockAssertionSoftly(keybag_handle_t handle, dispatch_block_t action);
//
// if you can't use the block version above, use these.
// !!!!!Remember to balance them!!!!!!
//
bool SecAKSKeybagDropLockAssertion(keybag_handle_t handle, CFErrorRef *error);
bool SecAKSKeybagHoldLockAssertion(keybag_handle_t handle, uint64_t timeout, CFErrorRef *error);


CFDataRef SecAKSCopyBackupBagWithSecret(size_t size, uint8_t *secret, CFErrorRef *error);

keyclass_t SecAKSSanitizedKeyclass(keyclass_t keyclass);

#endif
