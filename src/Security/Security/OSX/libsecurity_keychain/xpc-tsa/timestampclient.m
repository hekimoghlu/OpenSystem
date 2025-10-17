/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 15, 2023.
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
//  timestampclient.m
//  libsecurity_keychain
//

#import "timestampclient.h"

void sendTSARequest(CFDataRef tsaReq, const char *tsaURL, TSARequestCompletionBlock completionBlock)
{
    NSAutoreleasePool *pool = 	[[NSAutoreleasePool alloc] init];
    
    NSString *urlStr = [NSString stringWithCString:tsaURL encoding:NSUTF8StringEncoding];
    TimeStampClient *client = [[TimeStampClient alloc] initWithURLString:urlStr];
    [client post:(NSData *)tsaReq];
    [client start3:completionBlock];
    [client release];
    [pool release];
}

@implementation TimeStampClient

@synthesize delegate;
@synthesize url;
@synthesize urlRequest;
 
- (id)init
{
    if ((self = [super init])) {
    }
    return self;
}

- (id)initWithURLString:(NSString *)urlStr
{
    if ((self = [super init]))
    {
        NSString *escapedURLStr = [urlStr stringByAddingPercentEscapesUsingEncoding:NSUTF8StringEncoding];
        url = [[NSURL alloc] initWithString:escapedURLStr];
        urlRequest = [[NSMutableURLRequest alloc] initWithURL:self.url
            cachePolicy:NSURLRequestReloadIgnoringLocalCacheData timeoutInterval:(NSTimeInterval)15.0];
    }
    return self;
}
    
- (void)dealloc
{
    [url release];
    [urlRequest release];
    [super dealloc];
}

- (void)post: (NSData *)data
{
    NSMutableURLRequest *request = self.urlRequest;
    
    [request setHTTPMethod:@"POST"];
    [request setHTTPBody:data];
    [request setValue:@"application/timestamp-query" forHTTPHeaderField:@"Content-Type"];
}	

- (void)start3:(TSARequestCompletionBlock)completionBlock
{    
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_LOW, 0),
    ^{
        NSOperationQueue *opq = [[NSOperationQueue alloc] init];
        [NSURLConnection sendAsynchronousRequest:self.urlRequest
            queue:opq completionHandler: completionBlock            
        ];
    });
}

@end




