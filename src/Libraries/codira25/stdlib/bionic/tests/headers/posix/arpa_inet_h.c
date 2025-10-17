/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, August 2, 2022.
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
// Copyright (C) 2017 The Android Open Source Project
// SPDX-License-Identifier: BSD-2-Clause

#include <arpa/inet.h>

#include "header_checks.h"

static void arpa_inet_h() {
  TYPE(in_port_t);
  TYPE(in_addr_t);
  TYPE(struct in_addr);

  MACRO(INET_ADDRSTRLEN);
  MACRO(INET6_ADDRSTRLEN);

  FUNCTION(htonl, uint32_t (*f)(uint32_t));
  FUNCTION(htons, uint16_t (*f)(uint16_t));
  FUNCTION(ntohl, uint32_t (*f)(uint32_t));
  FUNCTION(ntohs, uint16_t (*f)(uint16_t));

  TYPE(uint32_t);
  TYPE(uint16_t);

  FUNCTION(inet_addr, in_addr_t (*f)(const char*));
}
