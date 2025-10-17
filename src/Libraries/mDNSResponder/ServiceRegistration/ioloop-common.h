/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, March 24, 2022.
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
#ifndef __IOLOOP_COMMON_H__
#define __IOLOOP_COMMON_H__

//======================================================================================================================
// MARK: - Functions

//======================================================================================================================
// service_connection_t methods.

/*!
 *  @brief
 *      Creates a special dnssd_txn_t to share its DNSServiceRef with all service_connection_t calls.
 *
 *  @result
 *      returns the created dnssd_txn_t if no error occurs, otherwise, NULL.
 */
#define dnssd_txn_create_shared() dnssd_txn_create_shared_(__FILE__, __LINE__)
dnssd_txn_t *NULLABLE
dnssd_txn_create_shared_(const char *const NONNULL file, const int line);

//======================================================================================================================

#endif // __IOLOOP_COMMON_H__
