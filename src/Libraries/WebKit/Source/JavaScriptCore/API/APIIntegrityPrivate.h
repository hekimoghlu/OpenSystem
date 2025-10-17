/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 19, 2024.
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
#ifndef JSIntegrityPrivate_h
#define JSIntegrityPrivate_h

#include <JavaScriptCore/JSContextRef.h>
#include <JavaScriptCore/JSObjectRef.h>
#include <JavaScriptCore/JSValueRef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
@function
@abstract Audits the integrity of the JSContextRef.
@param ctx The JSContext you want to audit.
@result JSContextRef that was passed in.
@discussion This function will crash if the audit detects any errors.
 */
JS_EXPORT JSContextRef jsAuditJSContextRef(JSContextRef ctx) JSC_API_AVAILABLE(macos(13.0), ios(16.1));

/*!
@function
@abstract Audits the integrity of the JSGlobalContextRef.
@param ctx The JSGlobalContextRef you want to audit.
@result JSGlobalContextRef that was passed in.
@discussion This function will crash if the audit detects any errors.
 */
JS_EXPORT JSGlobalContextRef jsAuditJSGlobalContextRef(JSGlobalContextRef ctx) JSC_API_AVAILABLE(macos(13.0), ios(16.1));

/*!
@function
@abstract Audits the integrity of the JSObjectRef.
@param obj The JSObjectRef you want to audit.
@result JSObjectRef that was passed in.
@discussion This function will crash if the audit detects any errors.
 */
JS_EXPORT JSObjectRef jsAuditJSObjectRef(JSObjectRef obj) JSC_API_AVAILABLE(macos(13.0), ios(16.1));

/*!
@function
@abstract Audits the integrity of the JSValueRef.
@param value The JSValueRef you want to audit.
@result JSValueRef that was passed in.
@discussion This function will crash if the audit detects any errors.
 */
JS_EXPORT JSValueRef jsAuditJSValueRef(JSValueRef value) JSC_API_AVAILABLE(macos(13.0), ios(16.1));

#ifdef __cplusplus
}
#endif

#endif /* JSIntegrityPrivate_h */
