/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 25, 2025.
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
/*!
 * @header
 * Definitions and interfaces for handling Cryptex1 nonces on Darwin platforms.
 */
#ifndef __IMAGE4_DARWIN_CRYPTEX1_NONCE_H
#define __IMAGE4_DARWIN_CRYPTEX1_NONCE_H

#include <os/base.h>
#include <stdint.h>

__BEGIN_DECLS
OS_ASSUME_NONNULL_BEGIN

/*!
 * @typedef darwin_cryptex1_nonce_t
 * A type describing nonce handle values for Cryptex1 nonce domains hosted on
 * Darwin.
 *
 * @const DARWIN_CRYPTEX1_NONCE_BOOT
 * The Cryptex1 boot nonce.
 *
 * @const DARWIN_CRYPTEX1_NONCE_ASSET_BRAIN
 * The Cryptex1 MobileAsset brain nonce.
 *
 * @const DARWIN_CRYPTEX1_NONCE_GENERIC
 * The Cryptex1 generic nonce.
 *
 * @const DARWIN_CRYPTEX1_NONCE_SIMULATOR_RUNTIME
 * The Cryptex1 simulator runtime nonce.
 *
 * @const DARWIN_CRYPTEX1_NONCE_MOBILE_ASSET_DFU
 * The Cryptex1 MobileAsset DFU nonce.
 */
OS_CLOSED_ENUM(darwin_cryptex1_nonce, uint32_t,
	DARWIN_CRYPTEX1_NONCE_BOOT = 1,
	DARWIN_CRYPTEX1_NONCE_ASSET_BRAIN = 2,
	DARWIN_CRYPTEX1_NONCE_GENERIC = 3,
	DARWIN_CRYPTEX1_NONCE_SIMULATOR_RUNTIME = 4,
	DARWIN_CRYPTEX1_NONCE_MOBILE_ASSET_DFU = 5,
	DARWIN_CRYPTEX1_NONCE_RESERVED_0 = 6,
	DARWIN_CRYPTEX1_NONCE_RESERVED_1 = 7,
	_DARWIN_CRYPTEX1_NONCE_CNT,
);

OS_ASSUME_NONNULL_END
__END_DECLS

#endif // __IMAGE4_DARWIN_CRYPTEX1_NONCE_H
