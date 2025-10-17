/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 21, 2023.
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
#import <PreferencePanes/PreferencePanes.h>
#import <SecurityInterface/SFAuthorizationView.h>
#import <dns_sd.h>

@class CNBonjourDomainView;
@class CNDomainBrowserView;

@interface DNSServiceDiscoveryPref : NSPreferencePane
{
    IBOutlet NSTextField          *hostName;
    IBOutlet NSTextField          *sharedSecretName;
    IBOutlet NSSecureTextField    *sharedSecretValue;
    IBOutlet NSTextField          *browseDomainTextField;
	IBOutlet NSTextField          *regDomainTextField;
	IBOutlet CNBonjourDomainView  *regDomainView;
    IBOutlet NSButton             *wideAreaCheckBox;
    IBOutlet NSButton             *hostNameSharedSecretButton;
	IBOutlet NSButton             *registrationSelectButton;
	IBOutlet NSButton             *registrationSharedSecretButton;
    IBOutlet NSButton             *applyButton;
    IBOutlet NSButton             *revertButton;
    IBOutlet NSWindow             *sharedSecretWindow;
	IBOutlet NSWindow             *addBrowseDomainWindow;
	IBOutlet NSWindow             *addBrowseDomainManualWindow;
	IBOutlet NSWindow             *selectRegistrationDomainWindow;
	IBOutlet NSWindow             *selectRegistrationDomainManualWindow;
    IBOutlet NSButton             *addBrowseDomainButton;
    IBOutlet NSButton             *removeBrowseDomainButton;
    IBOutlet NSButton             *secretOKButton;
    IBOutlet NSButton             *secretCancelButton;
    IBOutlet NSImageView          *statusImageView;
    IBOutlet NSTabView            *tabView;
	IBOutlet NSTableView          *browseDomainList;
	IBOutlet CNDomainBrowserView  *bonjourBrowserView;
	IBOutlet CNDomainBrowserView  *registrationBrowserView;
    IBOutlet SFAuthorizationView  *comboAuthButton;

    NSWindow            *mainWindow;
    NSString            *currentHostName;
    NSString            *currentRegDomain;
    NSArray             *currentBrowseDomainsArray;
    NSMutableArray      *browseDomainsArray;
    NSString            *defaultRegDomain;

    NSString            *hostNameSharedSecretName;
    NSString            *hostNameSharedSecretValue;
    NSString            *regSharedSecretName;
    NSString            *regSharedSecretValue;
    BOOL                currentWideAreaState;
    BOOL                prefsNeedUpdating;
    BOOL                browseDomainListEnabled;
    NSImage             *successImage;
    NSImage             *inprogressImage;
    NSImage             *failureImage;

    NSMutableArray      *registrationDataSource;
}

-(IBAction)applyClicked : (id)sender;
-(IBAction)enableBrowseDomainClicked : (id)sender;
-(IBAction)addBrowseDomainClicked : (id)sender;
-(IBAction)removeBrowseDomainClicked : (id)sender;
-(IBAction)revertClicked : (id)sender;
-(IBAction)changeButtonPressed : (id)sender;
-(IBAction)closeMyCustomSheet : (id)sender;
-(IBAction)wideAreaCheckBoxChanged : (id)sender;


-(NSMutableArray *)registrationDataSource;
-(NSString *)currentRegDomain;
-(NSArray *)currentBrowseDomainsArray;
-(NSString *)currentHostName;
-(NSString *)defaultRegDomain;
-(void)setDefaultRegDomain : (NSString *)domain;


-(void)enableApplyButton;
-(void)disableApplyButton;
-(void)applyCurrentState;
-(void)setupInitialValues;
-(void)toggleWideAreaBonjour : (BOOL)state;
-(void)updateApplyButtonState;
-(void)enableControls;
-(void)disableControls;
-(void)validateTextFields;
-(void)readPreferences;
-(void)savePreferences;
-(void)restorePreferences;
-(void)watchForPreferenceChanges;
-(void)updateStatusImageView;


-(NSString *)sharedSecretKeyName : (NSString * )domain;
-(NSString *)domainForHostName : (NSString *)hostNameString;
-(int)statusForHostName : (NSString * )domain;
-(NSData *)dataForDomainArray : (NSArray *)domainArray;
-(NSData *)dataForDomain : (NSString *)domainName isEnabled : (BOOL)enabled;
-(NSDictionary *)dictionaryForSharedSecret : (NSString *)secret domain : (NSString *)domainName key : (NSString *)keyName;
-(BOOL)domainAlreadyInList : (NSString *)domainString;
-(NSString *)trimCharactersFromDomain : (NSString *)domain;


// Delegate methods
-(void)authorizationViewDidAuthorize : (SFAuthorizationView *)view;
-(void)authorizationViewDidDeauthorize : (SFAuthorizationView *)view;
-(void)mainViewDidLoad;
-(void)controlTextDidChange : (NSNotification *) notification;

@end
