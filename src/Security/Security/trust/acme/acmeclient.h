/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, December 25, 2021.
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
//  acmeclient.h
//  XPCAcmeService
//

#import <Foundation/Foundation.h>

// See NSURLConnection completion handler
typedef void (^AcmeRequestCompletionBlock)(NSURLResponse *response, NSData *data, NSError *err);

void sendAcmeRequest(NSData *acmeReq, const char *acmeURL,
                     NSString *method, NSString *contentType,
                     AcmeRequestCompletionBlock completionBlock);

@interface AcmeClient : NSObject {
    __weak id delegate;
    NSURL *url;
    NSMutableURLRequest *urlRequest;
}
@property (weak) id delegate;
@property (retain) id url;
@property (retain) id urlRequest;

- (id)initWithURLString:(NSString *)urlStr;
- (void)post:(NSData *)data withMethod:(NSString *)method contentType:(NSString *)contentType;
- (void)start3:(AcmeRequestCompletionBlock)completionBlock;

@end
