/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, July 3, 2022.
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

// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef BSSL_PKI_IP_UTIL_H_
#define BSSL_PKI_IP_UTIL_H_

#include <openssl/base.h>

#include "input.h"

BSSL_NAMESPACE_BEGIN

inline constexpr size_t kIPv4AddressSize = 4;
inline constexpr size_t kIPv6AddressSize = 16;

// Returns whether `mask` is a valid netmask. I.e., whether it is the length of
// an IPv4 or IPv6 address, and is some number of ones, followed by some number
// of zeros.
OPENSSL_EXPORT bool IsValidNetmask(der::Input mask);

// Returns whether `addr1` and `addr2` are equal under the netmask `mask`.
OPENSSL_EXPORT bool IPAddressMatchesWithNetmask(der::Input addr1,
                                                der::Input addr2,
                                                der::Input mask);

BSSL_NAMESPACE_END

#endif  // BSSL_PKI_IP_UTIL_H_
