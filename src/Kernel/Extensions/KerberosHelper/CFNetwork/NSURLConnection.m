/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, May 21, 2022.
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
#import <KerberosHelper/NetworkAuthenticationHelper.h>
#import <KerberosHelper/KerberosHelper.h>
#import <Foundation/Foundation.h>
#import <Foundation/NSURLConnectionPrivate.h>
#import <CoreServices/CoreServices.h>
#import <CoreServices/CoreServicesPriv.h>

#include <err.h>

static NSString *username;
static NSString *password;

@interface NSURLCredential (NSURLCredentialInternal)
- (id) _initWithCFURLCredential:(CFURLCredentialRef)credential;
@end

@interface Foo : NSObject <NSURLConnectionDelegate> {
    NAHRef _nah;
    CFArrayRef _selections;
    CFIndex _selection_index;
}

- (void)req:(NSURL *)url;



- (BOOL)connection:(NSURLConnection *)connection canAuthenticateAgainstProtectionSpace:(NSURLProtectionSpace *)protectionSpace;
- (void)connection:(NSURLConnection *)connection didCancelAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge;
- (void)connection:(NSURLConnection *)connection didReceiveAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge;
- (BOOL)connectionShouldUseCredentialStorage:(NSURLConnection *)connection;

- (NSCachedURLResponse *)connection:(NSURLConnection *)connection willCacheResponse:(NSCachedURLResponse *)cachedResponse;
- (void)connection:(NSURLConnection *)connection didReceiveResponse:(NSURLResponse *)response;
- (void)connection:(NSURLConnection *)connection didReceiveData:(NSData *)data;
- (void)connection:(NSURLConnection *)connection didSendBodyData:(NSInteger)bytesWritten totalBytesWritten:(NSInteger)totalBytesWritten totalBytesExpectedToWrite:(NSInteger)totalBytesExpectedToWrite;
- (NSURLRequest *)connection:(NSURLConnection *)connection willSendRequest:(NSURLRequest *)request redirectResponse:(NSURLResponse *)redirectResponse;

- (void)connection:(NSURLConnection *)connection  didFailWithError:(NSError *)error;
- (void)connectionDidFinishLoading:(NSURLConnection *)connection;


@end

@implementation Foo


- (NSCachedURLResponse *)connection:(NSURLConnection *)connection willCacheResponse:(NSCachedURLResponse *)cachedResponse
{
	return nil;
}

- (void)connection:(NSURLConnection *)connection didCancelAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
}


- (void)connection:(NSURLConnection *)connection didReceiveData:(NSData *)data
{
}

- (void)connection:(NSURLConnection *)connection didSendBodyData:(NSInteger)bytesWritten totalBytesWritten:(NSInteger)totalBytesWritten totalBytesExpectedToWrite:(NSInteger)totalBytesExpectedToWrite
{
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
}
- (void)connectionDidFinishLoading:(NSURLConnection *)connection {
    NSLog(@"Finished...");
	exit(0);
}

- (void)connection:(NSURLConnection *)connection didFailWithError:(NSError *)error
{
	NSLog(@"didFailWithError");
}

- (NSURLRequest *)connection:(NSURLConnection *)connection willSendRequest:(NSURLRequest *)request redirectResponse:(NSURLResponse *)redirectResponse
{
	NSLog(@"willSendRequest");
	return request;
}

- (BOOL)connectionShouldUseCredentialStorage:(NSURLConnection *)connection
{
	NSLog(@"connectionShouldUseCredentialStorage");
	return NO;
}


- (void)connection:(NSURLConnection*)connection didReceiveAuthenticationChallenge:(NSURLAuthenticationChallenge *)challenge
{
    NSURLProtectionSpace *protectionSpace = [challenge protectionSpace];
	
	NSLog(@"didReceiveAuthenticationChallenge: %@ %@", [protectionSpace authenticationMethod], [protectionSpace host]);

    if (_nah == NULL) {
	CFMutableDictionaryRef info = NULL;
	CFDictionaryRef krbtoken = NULL;

	info = CFDictionaryCreateMutable(NULL, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);

	krbtoken = KRBCreateNegTokenLegacyKerberos(NULL);

	CFDictionaryAddValue(info, kNAHNegTokenInit, krbtoken);
	CFRelease(krbtoken);
	
	if (username){
	    NSLog(@"using %@ as username", username);
	    CFDictionaryAddValue(info, kNAHUserName, username);
	}
	
	if (password)
	    CFDictionaryAddValue(info, kNAHPassword, password);

	_nah = NAHCreate(NULL, (CFStringRef)[protectionSpace host], CFSTR("HTTP"), info);
	if (_nah == NULL)
	    goto failed;

	_selections = NAHGetSelections(_nah);
	_selection_index = 0;
    }

 next:
    if (_selection_index >= CFArrayGetCount(_selections))
	goto failed;

    NAHSelectionRef sel = (NAHSelectionRef)CFArrayGetValueAtIndex(_selections, _selection_index);

    _selection_index += 1;

    if (!NAHSelectionAcquireCredential(sel, NULL, NULL))
	goto next;


    CFStringRef clientPrincipal = NAHSelectionGetInfoForKey(sel, kNAHClientPrincipal);
    CFStringRef serverPrincipal = NAHSelectionGetInfoForKey(sel, kNAHServerPrincipal);

	NSLog(@"trying: client: %@ server: %@", (NSString *)clientPrincipal, (NSString *)serverPrincipal);
		  
	CFURLCredentialRef cfCredential = _CFURLCredentialCreateForKerberosTicket(NULL, clientPrincipal, serverPrincipal, NULL);
    if (cfCredential)
	goto failed;

    NSURLCredential *credential = [[[NSURLCredential alloc] _initWithCFURLCredential:cfCredential] autorelease];
	
    [[challenge sender] useCredential:credential forAuthenticationChallenge:challenge];

    CFRelease(cfCredential);

    return;

 failed:

    [[challenge sender] continueWithoutCredentialForAuthenticationChallenge:challenge];

    if (_selections)
	CFRelease(_selections);
    if (_nah)
	CFRelease(_nah);
    _nah = NULL;
    _selections = NULL;
}





- (void)req:(NSURL *)url
{
    NSURLRequest *request = [NSURLRequest requestWithURL:url];
    NSURLConnection *conn;

    conn = [[NSURLConnection alloc] initWithRequest: request delegate: self];

    [conn scheduleInRunLoop: [NSRunLoop currentRunLoop] forMode: NSDefaultRunLoopMode];

}

@end

int
main(int argc, char **argv)
{
    NSURL *url;
    Foo *foo;
    int ch;
    
    while ((ch = getopt(argc, argv, "u:p:")) != -1) {
	switch (ch) {
	    case 'u':
		username = [NSString stringWithCString:optarg encoding:NSUTF8StringEncoding];
		break;
	    case 'p':
		password = [NSString stringWithCString:optarg encoding:NSUTF8StringEncoding];
		break;
	}
    }
    
    argv += optind;
    argc -= optind;
    
    if (argc < 0)
	errx(1, "missing url");
	
    url = [NSURL URLWithString:[NSString stringWithUTF8String:argv[0]]];

    foo = [[Foo alloc] init];

    [foo req: url];

    [[NSRunLoop currentRunLoop] run];

    NSLog(@"done");

    return 0;
}
