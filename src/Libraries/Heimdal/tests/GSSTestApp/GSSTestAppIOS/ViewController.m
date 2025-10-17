/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, June 24, 2022.
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
//  GSSTestApp
//
//  Copyright (c) 2013 Apple, Inc. All rights reserved.
//


#include "ViewController.h"

#if !TARGET_OS_TV
#import <SafariServices/SafariServices.h>
#endif
#import "WebbyViewController.h"
#import "FakeXCTest.h"

#if !TARGET_OS_TV
@interface ViewController () <SFSafariViewControllerDelegate>
@property (strong) SFSafariViewController *safariViewController;
@end
#endif

@implementation ViewController

- (void)viewDidLoad
{
    [super viewDidLoad];
    self.credentialsTableController = [CredentialTableController getGlobalController];
    self.credentialsTableView.delegate = self.credentialsTableController;
    self.credentialsTableView.dataSource = self.credentialsTableController;
    self.credentialsTableView.allowsMultipleSelectionDuringEditing = NO;
}

- (void)viewDidAppear:(BOOL)animated {
    [self.credentialsTableController addNotification:self];
    [super viewDidAppear:animated];
}

- (void)viewDidDisappear:(BOOL)animated {
    [self.credentialsTableController removeNotification:self];
    [super viewDidDisappear:animated];
}

- (UIInterfaceOrientationMask)supportedInterfaceOrientations
{
    return UIInterfaceOrientationMaskPortrait;
}

- (void)GSSCredentialChangeNotifification {
    [self.credentialsTableView reloadData];
}

- (void)prepareForSegue:(UIStoryboardSegue *)segue sender:(id)sender {

    NSString *name = [segue identifier];

    if ([name isEqualToString:@"WKWebView"]) {
        WebbyViewController *vc = [segue destinationViewController];
        vc.type = [segue identifier];
    }
}

- (IBAction)safariViewController:(id)sender {
#if !TARGET_OS_TV
    self.safariViewController = [[SFSafariViewController alloc] initWithURL:[NSURL URLWithString:@"http://dc03.ads.apple.com/"]];

    [self presentViewController:self.safariViewController animated:YES completion:^{
        NSLog(@"presented");
    }];
#endif
}

@end
