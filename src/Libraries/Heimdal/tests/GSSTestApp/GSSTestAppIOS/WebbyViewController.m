/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, May 20, 2022.
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
//  WebbyViewController.m
//  GSSTestApp
//
//  Copyright (c) 2014 Apple, Inc. All rights reserved.
//

#import "WebbyViewController.h"

@interface WebbyViewController ()

@end

@implementation WebbyViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    [self.spinny setHidden:NO];

    self.urlTextField.text = @"http://dc03.ads.apple.com/";
    self.urlTextField.delegate = self;

    if ([self.type isEqualToString:@"WKWebView"]) {
        self.wk = [[WKWebView alloc] initWithFrame:[self.webbyView bounds]];
        self.wk.navigationDelegate = self;

        [self.webbyView addSubview:self.wk];
    } else {
        abort();
    }

    [self gotoURL:self.urlTextField.text];
}

- (IBAction)reload:(id)sender {
    if (self.wk)
        [self.wk reload];
}

- (IBAction)goBack:(id)sender {
    if (self.wk)
        [self.wk goBack];
}

- (void)startLoading {
    [self.spinny startAnimating];
    [self.reloadButton setHidden:YES];
    [self.spinny setHidden:NO];
}

- (void)doneLoading {
    [self.spinny stopAnimating];
    [self.reloadButton setHidden:NO];
    [self.spinny setHidden:YES];
}

#pragma mark - WKWebView

- (void)webView:(WKWebView *)webView didStartProvisionalNavigation:(WKNavigation *)navigation
{
    [self startLoading];
    self.urlTextField.text = [[self.wk URL] absoluteString];
}

- (void)webView:(WKWebView *)webView didFinishNavigation:(WKNavigation *)navigation
{
    [self doneLoading];
}

- (void)webView:(WKWebView *)webView didFailNavigation:(WKNavigation *)navigation withError:(NSError *)error
{
    [self doneLoading];
}

- (void)gotoURL:(NSString *)url {
    NSURLRequest *request = [NSURLRequest requestWithURL:[NSURL URLWithString:url]];

    if (self.wk)
        [self.wk loadRequest:request];
}

- (BOOL) textFieldShouldReturn:(UITextField *)textField{

    [textField resignFirstResponder];
    [self gotoURL:textField.text];
    return YES;
}

@end
