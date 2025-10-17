/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, March 31, 2023.
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
#ifndef SecProtocolConfiguration_h
#define SecProtocolConfiguration_h

#include <Security/SecProtocolObject.h>
#include <Security/SecureTransport.h>

#include <dispatch/dispatch.h>
#include <xpc/xpc.h>

#ifndef SEC_OBJECT_IMPL
/*!
 * A `sec_protocol_configuration` is an object that encapsulates App Transport Security
 * information and vends `sec_protocol_options` to clients for creating new connections.
 * It may also be queried to determine for what domains TLS is required.
 */
SEC_OBJECT_DECL(sec_protocol_configuration);
#endif // !SEC_OBJECT_IMPL

__BEGIN_DECLS

SEC_ASSUME_NONNULL_BEGIN

/*!
 * @function sec_protocol_configuration_copy_singleton
 *
 * @abstract
 *      Copy the per-process `sec_protocol_configuration_t` object.
 *
 * @return A non-nil `sec_protocol_configuration_t` instance.
 */
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0))
SEC_RETURNS_RETAINED sec_protocol_configuration_t
sec_protocol_configuration_copy_singleton(void);

/*!
 * @function sec_protocol_configuration_set_ats_overrides
 *
 * @abstract
 *      Set ATS overrides
 *
 * @param config
 *      A `sec_protocol_configuration_t` instance.
 *
 * @param override_dictionary
 *      A `CFDictionaryRef` dictionary containing the ATS overrides as
 *      documented here: https://developer.apple.com/library/archive/documentation/General/Reference/InfoPlistKeyReference/Articles/CocoaKeys.html#//apple_ref/doc/uid/TP40009251-SW33
 *
 * @return True if successful, and false otherwise.
 */
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0))
bool
sec_protocol_configuration_set_ats_overrides(sec_protocol_configuration_t config, CFDictionaryRef override_dictionary);

/*!
 * @function sec_protocol_configuration_copy_transformed_options
 *
 * @abstract
 *      Transform an existing `sec_protocol_options_t` instance with a `sec_protocol_configuration_t` instance.
 *
 * @param config
 *      A `sec_protocol_configuration_t` instance.
 *
 * @param options
 *      A `sec_protocol_options_t` instance.
 *
 * @return The transformed `sec_protocol_options` instance.
 */
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0))
SEC_RETURNS_RETAINED __nullable sec_protocol_options_t
sec_protocol_configuration_copy_transformed_options(sec_protocol_configuration_t config, sec_protocol_options_t options);

/*!
 * @function sec_protocol_configuration_copy_transformed_options_for_host
 *
 * @abstract
 *      Transform an existing `sec_protocol_options_t` instance with a `sec_protocol_configuration_t` instance
 *      using a specific host endpoint. Note that the service (port) is omitted from this formula.
 *
 * @param options
 *      A `sec_protocol_options_t` instance.
 *
 * @param host
 *      A NULL-terminated C string containing the host in question.
 *
 * @param is_direct
 *      A flag which indicates if the given hostname is local (direct).
 *
 * @return The transformed `sec_protocol_options` instance.
 */
API_AVAILABLE(macos(15.4), ios(18.4), watchos(11.4), tvos(18.4), visionos(2.4))
SEC_RETURNS_RETAINED __nullable sec_protocol_options_t
sec_protocol_configuration_copy_transformed_options_for_host(sec_protocol_options_t options, const char *host, bool is_direct);

/*!
 * @function sec_protocol_configuration_copy_transformed_options_for_address
 *
 * @abstract
 *      Transform an existing `sec_protocol_options_t` instance with a `sec_protocol_configuration_t` instance
 *      using a specific host endpoint. Note that the service (port) is omitted from this formula.
 *
 * @param options
 *      A `sec_protocol_options_t` instance.
 *
 * @param address
 *      A NULL-terminated C string containing the address in question.
 *
 * @param is_direct
 *      A flag which indicates if the given address is local (direct).
 *
 * @return The transformed `sec_protocol_options` instance.
 */
API_AVAILABLE(macos(15.4), ios(18.4), watchos(11.4), tvos(18.4), visionos(2.4))
SEC_RETURNS_RETAINED __nullable sec_protocol_options_t
sec_protocol_configuration_copy_transformed_options_for_address(sec_protocol_options_t options, const char *address, bool is_direct);

/*!
 * @function sec_protocol_configuration_tls_required
 *
 * @abstract
 *      Determine if TLS is required by policy for a generic connection. Note that the service (port) is omitted
 *      from this formula.
 *
 * @param config
 *      A `sec_protocol_configuration_t` instance.
 *
 * @return True if connections require TLS, and false otherwise.
 */
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0))
bool
sec_protocol_configuration_tls_required(sec_protocol_configuration_t config);

/*!
 * @function sec_protocol_configuration_tls_required_for_host
 *
 * @abstract
 *      Determine if TLS is required -- by policy -- for the given host endpoint. Note that the service (port) is
 *      omitted from this formula.
 *
 * @param config
 *      A `sec_protocol_configuration_t` instance.
 *
 * @param host
 *      A NULL-terminated C string containing the host endpoint to examine.
 *
 * @param is_direct
 *      A flag which indicates if the given hostname is local (direct).
 *
 * @return True if connections to the endpoint require TLS, and false otherwise.
 */
API_AVAILABLE(macos(10.15), ios(13.0), watchos(6.0), tvos(13.0))
bool
sec_protocol_configuration_tls_required_for_host(sec_protocol_configuration_t config, const char *host, bool is_direct);

/*!
 * @function sec_protocol_configuration_tls_required_for_address
 *
 * @abstract
 *      Determine if TLS is required -- by policy -- for the given address endpoint.
 *
 * @param config
 *      A `sec_protocol_configuration_t` instance.
 *
 * @param address
 *      A NULL-terminated C string containing the address endpoint to examine.
 *
 * @return True if connections to the endpoint require TLS, and false otherwise.
 */
API_AVAILABLE(macos(15.4), ios(18.4), watchos(11.4), tvos(18.4), visionos(2.4))
bool
sec_protocol_configuration_tls_required_for_address(sec_protocol_configuration_t config, const char *address, bool is_direct);

#define SEC_PROTOCOL_HAS_APP_TRANSPORT_SECURITY_SUPPORT

SEC_ASSUME_NONNULL_END

__END_DECLS

#endif // SecProtocolConfiguration_h
