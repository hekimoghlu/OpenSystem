/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, November 11, 2023.
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

// Copyright (c) 2007 Apple Inc. All Rights Reserved.

#import <CoreFoundation/CoreFoundation.h>
#import <DirectoryService/DirectoryService.h>

// find user records by attribute

bool ds_open_search_node(tDirReference _directoryRef, tDirNodeReference *_nodeRef);

CF_RETURNS_RETAINED
CFMutableArrayRef ds_find_user_records_by_dsattr(tDirReference _directoryRef, tDirNodeReference _nodeRef, const char *attr, const char *value);

// authenticate user and get info

bool ds_open_node_for_user_record(tDirReference _directoryRef, const char *name, tDirNodeReference *_nodeRef);

tDirStatus ds_dir_node_auth_operation(tDirReference _directoryRef, tDirNodeReference _nodeRef, const char *operation, const char *username, const char *password, bool authentication_only);

CF_RETURNS_RETAINED
CFMutableArrayRef ds_get_user_records_for_name(tDirReference dir_ref, tDirNodeReference node_ref, const char *name, uint32_t max_records);

tDirStatus ds_set_attribute_in_user_record(tDirReference _directoryRef, tDirNodeReference _nodeRef, const char *recordname, size_t length, const void *data, const char *attr_type, uint32_t value_index);

CFMutableArrayRef find_user_record_by_attr_value(const char *attr, const char *value);
