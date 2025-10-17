/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, September 12, 2022.
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
// *****************************************************************************
//  cclkeys.h
//
//  Created by kevine on 3/1/06.
//  Recreated by sspies 15 January 2007
// *****************************************************************************


#ifndef CCLKEYS_H
#define CCLKEYS_H

#include <SystemConfiguration/SCSchemaDefinitions.h>

// CCL bundles ...
#define kCCLFileExtension       CFSTR("ccl")

// top-level structure for bundle dictionary
#define kCCLPersonalitiesKey    CFSTR("CCL Personalities")      // dict
// personality names are keys; -> kSCPropNetModemConnectionPersonality
#define kCCLDefaultPersonalityKey CFSTR("Default Personality")  // dict
#define kCCLVersionKey          CFSTR("CCL Version")            // integer
#define kCCLBundleVersion       1

// Personality's type
#define kCCLConnectTypeKey      CFSTR("Connect Type")       // string
#define kCCLConnectGPRS             CFSTR("GPRS")
#define kCCLConnectDialup           CFSTR("Dialup")

// flat scripts that this personality obsoletes
#define kCCLSupersedesKey    CFSTR("Supersedes")            // array of str

// How personality is described in the UI
#define kCCLDeviceNamesKey      CFSTR("Device Names")       // array
#define kCCLVendorKey  kSCPropNetModemDeviceVendor /*("DeviceVendor")*/ // str
#define kCCLModelKey   kSCPropNetModemDeviceModel  /*("DeviceModel")*/  // str

// Device capabilities assumed by personality
#define kCCLGPRSCapabilitiesKey CFSTR("GPRS Capabilities")  // dict
#define kCCLSupportsCIDQueryKey     CFSTR("CID Query")         // bool
#define kCCLSupportsDataModeKey     CFSTR("Data Mode")         //bool(AT+CGDATA)
#define kCCLSupportsDialModeKey     CFSTR("Dial Mode")         // bool (ATD *99)
#define kCCLMaximumCIDKey           CFSTR("Maximum CID")       // integer
#define kCCLIndependentCIDs         CFSTR("Independent CIDs")  // bool
#define kCCLIndependentCIDsKey      CFSTR("Independent CIDs")  // bool
// Independent CIDs means that commands like AT+CGDCONT= won't override
// APN valumes stored (by CID) in the device.

// Parameters passed to the script for this personality
#define kCCLScriptNameKey       CFSTR("Script Name")        // in Resources/
#define kCCLParametersKey       CFSTR("CCLParameters")      // dict
#define kCCLConnectSpeedKey         CFSTR("Connect Speed")     // string (^20)
#define kCCLInitStringKey           CFSTR("Init String")       // string (^21)
#define kCCLPreferredAPNKey         CFSTR("Preferred APN")     // str (-> ^22)
#define kCCLPreferredCIDKey         CFSTR("Preferred CID")     // int (-> ^23)
// Preferred CID w/o Preferred APN means use APN stored "at" that CID in phone
// A Preferred CID with Preferred APN means assign the given APN to said CID

// varStrings 23-26 reserved for future language-defined arguments

// Four script-defined arguments to be used as seen fit
#define kCCLVarString27Key          CFSTR("varString 27")   // string (^27)
#define kCCLVarString28Key          CFSTR("varString 28")   // string (^28)
#define kCCLVarString29Key          CFSTR("varString 29")   // string (^29)
#define kCCLVarString30Key          CFSTR("varString 30")   // string (^30)

// traditional argument now in the dict passed from pppd to CCLEngine
#define kModemPhoneNumberKey    CFSTR("Phone Number")   // string (^1 & ^7-9)    


// CCLEngine control keys
#define kCCLEngineDictKey           CFSTR("Engine Control") // control dict

// engine control parameters
#define kCCLEngineModeKey               CFSTR("Mode")               // str
#define kCCLEngineModeConnect           CFSTR("Connect")
#define kCCLEngineModeDisconnect        CFSTR("Disconnect")
#define kCCLEngineBundlePathKey         CFSTR("Bundle Path")        // str
#define kCCLEngineServiceIDKey          CFSTR("Service ID")         // str
#define kCCLEngineAlertNameKey          CFSTR("Alert Name")         // str
#define kCCLEngineIconPathKey           CFSTR("Icon Path")          // str
#define kCCLEngineCancelNameKey         CFSTR("Cancel Name")        // str
#define kCCLEngineSyslogLevelKey        CFSTR("Syslog Level")       // int
#define kCCLEngineSyslogFacilityKey     CFSTR("Syslog Facility")    // int
#define kCCLEngineVerboseLoggingKey     CFSTR("Verbose Logging")    // int
#define kCCLEngineLogToStdErrKey        CFSTR("Log To Stderr")      // int

// #define kCCLEngineLogFileKey         CFSTR("Log File")           // unused
// #define kCCLEngineBundleIconURLKey      CFSTR("BundleIconURL")   // unused

#endif      // CCLKEYS_H
