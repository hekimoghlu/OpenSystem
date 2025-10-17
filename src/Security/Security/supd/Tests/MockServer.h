/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, December 18, 2023.
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



#import <Foundation/Foundation.h>
#import <CFNetwork/CFNetwork.h>
#import <CFNetwork/CFNetworkPriv.h>

@class HTTPRequest;
@class HTTPResponse;
@class HTTPConnection;
@class HTTPServer;

NS_ASSUME_NONNULL_BEGIN

extern const uint16_t kHTTPServerPortAny;

@protocol HTTPServerProtocol <NSObject>
- (HTTPConnection *_Nullable)newConnection:(_CFHTTPServerConnectionRef)connection server:(HTTPServer *)server ;
@end

@interface HTTPServer: NSObject
@property id<HTTPServerProtocol> acceptConnection;

- (instancetype)initWithName:(NSString *)serviceName type:(NSString *)serviceType port:(uint16_t)servicePort;
- (void)start;
- (void)stop:(BOOL)wait;

@property (readonly) uint16_t port;
@end

@interface HTTPRequest : NSObject
@property (nonatomic, readonly) NSURL *URL;
@property (nonatomic, readonly) NSString *httpVersion;
@property (nonatomic, readonly) NSString *method;
@property (nonatomic, readonly) NSString *path;
@property (nonatomic, readonly) NSDictionary *headers;
@property (nonatomic, readonly) NSDictionary *cookies;

@end


@interface HTTPConnection : NSObject
- (void)didBecomeInvalid;
- (void)didReceiveError:(NSError *)error;
- (void)didReceiveRequest:(HTTPRequest *)request;
- (void)didSendResponse:(HTTPResponse *)response forRequest:(HTTPRequest *)request;
- (void)didFailToSendResponse:(HTTPResponse *)response forRequest:(HTTPRequest *)request;

- (void)enqueueResponse:(HTTPResponse *)response;

- (void)requestData:(HTTPRequest *)request withCompletion:(void (^)(NSError * _Nullable, NSData * _Nullable))completion;

- (instancetype)init NS_UNAVAILABLE;
- (instancetype)initWithConnectionRef:(_CFHTTPServerConnectionRef)connection server:(HTTPServer *)server;

@end

@interface HTTPResponse : NSObject
@property (nonatomic, readonly) NSUInteger code;
@property (nonatomic, readonly) NSData *data;
@property (nonatomic, readonly) HTTPRequest *request;
@property (nonatomic, readonly) NSDictionary *headers;

- (id)initWithRequest:(HTTPRequest *)request code:(NSUInteger)code data:(NSData *)data;
- (void)addHeaderValue:(NSString *)value forKey:(NSString *)key;
- (BOOL)hasData;
@end

NS_ASSUME_NONNULL_END
