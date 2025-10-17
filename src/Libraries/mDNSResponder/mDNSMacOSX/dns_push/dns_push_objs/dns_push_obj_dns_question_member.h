/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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
#ifndef DNS_PUSH_OBJ_DNS_QUESTION_MEMBER_H
#define DNS_PUSH_OBJ_DNS_QUESTION_MEMBER_H

#include "mDNSFeatures.h"
#if MDNSRESPONDER_SUPPORTS(APPLE, DNS_PUSH)

//======================================================================================================================
// MARK: - Headers

#include "dns_push_obj.h"
#include "dns_common.h"
#include <stdint.h>
#include <stdbool.h>

#include "nullability.h"

//======================================================================================================================
// MARK: - Object Reference Definition

DNS_PUSH_OBJECT_TYPEDEF_OPAQUE_POINTER(dns_question_member);
DNS_PUSH_OBJECT_TYPEDEF_OPAQUE_POINTER(context);

//======================================================================================================================
// MARK: - Object Methods

dns_push_obj_dns_question_member_t NULLABLE
dns_push_obj_dns_question_member_create(dns_obj_error_t * NULLABLE out_error);

void
dns_push_obj_dns_question_member_set_context(dns_push_obj_dns_question_member_t NONNULL member,
	dns_push_obj_context_t NONNULL context);

dns_push_obj_context_t NULLABLE
dns_push_obj_dns_question_member_get_context(dns_push_obj_dns_question_member_t NONNULL member);

#endif // MDNSRESPONDER_SUPPORTS(APPLE, DNS_PUSH)

#endif // DNS_PUSH_OBJ_DNS_QUESTION_MEMBER_H
