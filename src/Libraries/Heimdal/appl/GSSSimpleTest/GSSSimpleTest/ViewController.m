/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 5, 2021.
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
//  ViewController.m
//  GSSSimpleTest
//
//  Copyright (c) 2013 Apple. All rights reserved.
//

#import "ViewController.h"
#import <Foundation/Foundation.h>

@interface ViewController () <NSURLConnectionDelegate>
@property (retain) NSURL *baseURL;
@property (retain) NSMutableData *content;
@property (retain) NSOperationQueue *opQueue;
@property (retain) NSURLResponse *response;
@property (retain) NSURLConnection *conn;
@end

@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
    self.opQueue = [[NSOperationQueue alloc] init];
}

#pragma mark HTTP test

- (void)connection:(NSURLConnection *)connection didReceiveData:(NSData *)data
{
    [self.content appendData:data];
}

- (BOOL)connection:(NSURLConnection *)connection canAuthenticateAgainstProtectionSpace:(NSURLProtectionSpace *)protectionSpace
{
    NSLog(@"canAuthenticateAgainstProtectionSpace: %@", [protectionSpace authenticationMethod]);
    
    if ([[protectionSpace authenticationMethod] isEqualToString:NSURLAuthenticationMethodNegotiate])
        return YES;
    
    return NO;
}

- (void)connection:(NSURLConnection *)connection didReceiveResponse:(NSURLResponse *)response {
    NSLog(@"Connection didReceiveResponse! Response - %@", response);
    self.response = response;
}

- (void)connectionDidFinishLoading:(NSURLConnection *)connection {
    
    __block NSString *html = [[NSString alloc] initWithData:self.content encoding:NSUTF8StringEncoding];
    __block NSString *status;
    

    self.content = NULL;
    if ([self.response isKindOfClass:[NSHTTPURLResponse class]]) {
        NSHTTPURLResponse *urlResponse = (NSHTTPURLResponse *)self.response;
        status = [NSString stringWithFormat:@"complete with status: %d", (int)[urlResponse statusCode]];
    } else {
        status = [NSString stringWithFormat:@"complete"];
    }
    NSLog(@"data: %@", html);
    dispatch_async(dispatch_get_main_queue(), ^{
        [self.result loadHTMLString:html baseURL:self.baseURL];
    });
}

- (void)connection:(NSURLConnection *)connection didFailWithError:(NSError *)error
{
	NSLog(@"didFailWithError");
    dispatch_async(dispatch_get_main_queue(), ^{
        [self.result loadHTMLString:@"failed" baseURL:nil];
    });
}

- (NSURLRequest *)connection:(NSURLConnection *)connection willSendRequest:(NSURLRequest *)request redirectResponse:(NSURLResponse *)redirectResponse
{
	NSLog(@"willSendRequest");
	return request;
}

- (BOOL)connectionShouldUseCredentialStorage:(NSURLConnection *)connection
{
	NSLog(@"connectionShouldUseCredentialStorage");
	return YES;
}


- (void)connection:(NSURLConnection *)connection willSendRequestForAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge {

    NSURLProtectionSpace *protectionSpace = [challenge protectionSpace];
	
	NSLog(@"didReceiveAuthenticationChallenge: %@ %@", [protectionSpace authenticationMethod], [protectionSpace host]);
    
    [[challenge sender] performDefaultHandlingForAuthenticationChallenge:challenge];
}

- (IBAction)checkURL:(id)sender {
    
    [self.url resignFirstResponder];

    self.baseURL = [NSURL URLWithString:[self.url text]];
    
    NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:self.baseURL];
    
    [request setCachePolicy:NSURLRequestReloadIgnoringCacheData];
    
    self.conn = [[NSURLConnection alloc] initWithRequest: request delegate: self startImmediately:NO];
    self.content = [NSMutableData data];
    
    [self.result loadHTMLString:@"<html><body>performing test</body></html>" baseURL:nil];

    [self.conn setDelegateQueue:self.opQueue];
    [self.conn start];
}


@end
