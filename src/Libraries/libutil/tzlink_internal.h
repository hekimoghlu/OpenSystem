/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, November 11, 2021.
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
#define TZLINK_SERVICE_NAME "com.apple.tzlink"

#define TZLINK_ENTITLEMENT "com.apple.tzlink.allow"

#define TZLINK_KEY_REQUEST_TIMEZONE "tz" // string
#define TZLINK_KEY_REPLY_ERROR "error" // uint64

// These also probably don't need to be here
#define KERN_APFSPREBOOTUUID "kern.apfsprebootuuid"
#define KERN_BOOTUUID "kern.bootuuid"

// These could probably also be cleaned up since they're not used
int get_preboot_volume_uuid(char **uuid_out);
const char *construct_preboot_volume_path(const char *format_string, const char *uuid);
int check_update_timezone_db(char *preboot_volume_uuid);
int update_preboot_volume(const char *localtime_target_path);
int get_tz_version(const char *tz_path, char **tz_version);
char *file_path_append(const char *trunk, const char *suffix);
int remove_files(const char* path);
