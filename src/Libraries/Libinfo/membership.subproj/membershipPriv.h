/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, July 29, 2024.
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
#ifndef _MEMBERSHIPPRIV_H_
#define _MEMBERSHIPPRIV_H_

#include <uuid/uuid.h>
#include <ntsid.h>
#include <os/availability.h>

#define MBR_UU_STRING_SIZE 37
#define MBR_MAX_SID_STRING_SIZE 200

#define SID_TYPE_USER 0
#define SID_TYPE_GROUP 1

#define MBR_REC_TYPE_USER 1
#define MBR_REC_TYPE_GROUP 2

/* only supported by mbr_identifier_translate for target type */
#define ID_TYPE_UID_OR_GID 30
#define ID_TYPE_NAME 31
#define ID_TYPE_WINDOWS_FQN 32

__BEGIN_DECLS

int mbr_reset_cache();
int mbr_user_name_to_uuid(const char *name, uuid_t uu);
int mbr_group_name_to_uuid(const char *name, uuid_t uu);
int mbr_check_membership_by_id(uuid_t user, gid_t group, int *ismember);
int mbr_check_membership_refresh(const uuid_t user, uuid_t group, int *ismember);

/* mbr_uuid_to_string should use uuid_unparse from uuid.h */
int mbr_uuid_to_string(const uuid_t uu, char *string) API_DEPRECATED("No longer supported", macos(10.4, 10.8));

/* mbr_string_to_uuid should use uuid_parse from uuid.h */
int mbr_string_to_uuid(const char *string, uuid_t uu) API_DEPRECATED("No longer supported", macos(10.4, 10.8));

int mbr_uuid_to_sid_type(const uuid_t uu, nt_sid_t *sid, int *id_type);
int mbr_set_identifier_ttl(int id_type, const void *identifier, size_t identifier_size, unsigned int seconds);

/* new SPI to allow translation from any-to-any type, pass ID_TYPE_UID_OR_GID when translating to a UID */
int mbr_identifier_translate(int id_type, const void *identifier, size_t identifier_size, int target_type, void **result, int *rec_type);

/* 
 * groupid_type does not support ID_TYPE_GSS_EXPORT_NAME
 */
int mbr_check_membership_ext(int userid_type, const void *userid, size_t userid_size, int groupid_type, const void *groupid, int refresh, int *isMember);

SPI_AVAILABLE(macos(12.0), ios(15.0), watchos(8.0), tvos(15.0))
int mbr_close_connections();

__END_DECLS

#endif /* !_MEMBERSHIPPRIV_H_ */
