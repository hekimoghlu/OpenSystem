/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 7, 2023.
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
//
//  HTTPStatusCodes.h
//  utilities
//

#ifndef _HTTPSTATUSCODES_H_
#define _HTTPSTATUSCODES_H_

#pragma mark RFC 2616 (Hypertext Transfer Protocol -- HTTP/1.1)
// http://tools.ietf.org/html/rfc2616#section-10

#pragma mark 1xx - Informational
#define HTTPResponseCodeContinue                      100
#define HTTPResponseCodeSwitchingProtocols            101

#pragma mark 2xx - Successful
#define HTTPResponseCodeOK                            200
#define HTTPResponseCodeCreated                       201
#define HTTPResponseCodeAccepted                      202
#define HTTPResponseCodeNonAuthoritativeInformation   203
#define HTTPResponseCodeNoContent                     204
#define HTTPResponseCodeResetContent                  205
#define HTTPResponseCodePartialContent                206

#pragma mark 3xx - Redirection
#define HTTPResponseCodeMultipleChoices               300
#define HTTPResponseCodeMovedPermanently              301
#define HTTPResponseCodeFound                         302
#define HTTPResponseCodeSeeOther                      303
#define HTTPResponseCodeNotModified                   304
#define HTTPResponseCodeUseProxy                      305
#define HTTPResponseCodeTemporaryRedirect             307

#pragma mark 4xx - Client Error
#define HTTPResponseCodeBadRequest                    400
#define HTTPResponseCodeUnauthorized                  401
#define HTTPResponseCodePaymentRequired               402
#define HTTPResponseCodeForbidden                     403
#define HTTPResponseCodeNotFound                      404
#define HTTPResponseCodeMethodNotAllowed              405
#define HTTPResponseCodeNotAcceptable                 406
#define HTTPResponseCodeProxyAuthenticationRequired   407
#define HTTPResponseCodeRequestTimeout                408
#define HTTPResponseCodeConflict                      409
#define HTTPResponseCodeGone                          410
#define HTTPResponseCodeLengthRequired                411
#define HTTPResponseCodePreconditionFailed            412
#define HTTPResponseCodeRequestEntityTooLarge         413
#define HTTPResponseCodeRequestURITooLong             414
#define HTTPResponseCodeUnsupportedMediaType          415
#define HTTPResponseCodeRequestedRangeNotSatisfiable  416
#define HTTPResponseCodeExpectationFailed             417
#define HTTPResponseCodeImATeaPot                     418

#pragma mark 5xx - Server Error
#define HTTPResponseCodeInternalServerError           500
#define HTTPResponseCodeNotImplemented                501
#define HTTPResponseCodeBadGateway                    502
#define HTTPResponseCodeServiceUnavailable            503
#define HTTPResponseCodeGatewayTimeout                504
#define HTTPResponseCodeHTTPVersionNotSupported       505

#pragma mark -
#pragma mark RFC 4918 (HTTP Extensions for Web Distributed Authoring and Versioning (WebDAV))
// http://tools.ietf.org/html/rfc4918#section-11

#pragma mark 2xx - Successful
#define HTTPResponseCodeMultiStatus                   207

#pragma mark 4xx - Client Error
#define HTTPResponseCodeUnprocessableEntity           422
#define HTTPResponseCodeLocked                        423
#define HTTPResponseCodeFailedDependency              424

#pragma mark 5xx - Server Error
#define HTTPResponseCodeInsufficientStorage           507

#endif
