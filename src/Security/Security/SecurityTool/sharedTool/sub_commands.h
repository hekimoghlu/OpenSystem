/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 27, 2025.
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
// This file can't be once, it gets included multiple times to get definitions and declarations.

#include "SecurityTool/sharedTool/SecurityCommands.h"

#if !TARGET_OS_BRIDGE
#include "keychain/SecureObjectSync/Tool/keychain_sync.h"
#include "keychain/SecureObjectSync/Tool/keychain_sync_test.h"
#include "keychain/SecureObjectSync/Tool/keychain_log.h"
#include "keychain/SecureObjectSync/Tool/recovery_key.h"
#endif
