/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, April 1, 2022.
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
/*
   File:      MDSAttrParser.h

   Contains:  Classes to parse XML plists and fill in MDS DBs with the
              attributes found there.  

   Copyright (c) 2001,2011,2014 Apple Inc. All Rights Reserved.
*/

#ifndef _MDS_ATTR_PARSER_H_
#define _MDS_ATTR_PARSER_H_  1

#include <Security/cssmtype.h>
#include "MDSSession.h"
#include "MDSDictionary.h"
#include "MDSAttrStrings.h"
#include <CoreFoundation/CoreFoundation.h>

/*
 * Hard-coded strings, which we attempt to keep to a minimum
 */
 
/* extension of a bundle's MDS files */
#define MDS_INFO_TYPE				"mdsinfo"

/* key in an MDS info file determining whether it's for CSSM, plugin, or
 * Plugin-specific MDS record type */
#define MDS_INFO_FILE_TYPE			"MdsFileType"

/* Values for MDS_INFO_FILE_TYPE */
#define MDS_INFO_FILE_TYPE_CSSM		"CSSM"
#define MDS_INFO_FILE_TYPE_PLUGIN	"PluginCommon"
#define MDS_INFO_FILE_TYPE_RECORD	"PluginSpecific"
 
/* For MDS_INFO_FILE_TYPE_RECORD files, this key is used to find the 
 * CSSM_DB_RECORDTYPE associated with the file's info. */
#define MDS_INFO_FILE_RECORD_TYPE	"MdsRecordType"

/* key for file description string, for debugging and documentation (since 
 * PropertyListEditor does not support comments) */
#define MDS_INFO_FILE_DESC			"MdsFileDescription"


namespace Security
{

/*
 * The purpose of the MDSAttrParser class is to process a set of plist files
 * in a specified bundle or framework, parsing them to create data which 
 * is written to a pair of open DBs. Each plist file represents the bundle's
 * entries for one or more MDS relations. Typically a bundle will have 
 * multiple plist files. 
 */

/* base class for all parsers */
class MDSAttrParser
{
public:
	MDSAttrParser(
		const char *bundlePath,
		MDSSession &dl,
		CSSM_DB_HANDLE objectHand,
		CSSM_DB_HANDLE cdsaDirHand);
	virtual ~MDSAttrParser();
	
	/* the bulk of the work */
	void parseAttrs(CFStringRef subdir = NULL);
	
	/* parse a single file, by path URL -- throws on parse error */
	void parseFile(CFURLRef theFileUrl, CFStringRef subdir = NULL);
	
	void setDefaults(const MDS_InstallDefaults *defaults) { mDefaults = defaults; }
	
	const char *guid()  { return mGuid; }
	
private:
	void logFileError(
		const char *op,
		CFURLRef file,	
		CFStringRef errStr,		// optional if you have it
		SInt32 *errNo);			// optional if you have it
		
	/*
	 * Parse a CSSM info file.
	 */
	void parseCssmInfo(
		MDSDictionary *theDict);
		
	/*
	 * Parse a Plugin Common info file.
	 */
	void parsePluginCommon(
		MDSDictionary *theDict);
		
	/*
	 * Parse a Plugin-specific file.
	 */
	void parsePluginSpecific(
		MDSDictionary *theDict);
		
	/*
	 * Given an open dictionary (representing a parsed XML file), create
	 * an MDS_OBJECT_RECORDTYPE record and add it to mObjectHand. This is
	 * used by both parseCssmInfo and parsePluginCommon.
	 */
	void parseObjectRecord(
		MDSDictionary *dict);
		
	/*
	 * Given an open dictionary and a RelationInfo defining a schema, fetch all
	 * attributes associated with the specified schema from the dictionary
	 * and write them to specified DB.
	 */
	void parseMdsRecord(
		MDSDictionary	 			*mdsDict,
		const RelationInfo 			*relInfo,
		CSSM_DB_HANDLE				dbHand);

	/*
	 * Special case handlers for MDS_CDSADIR_CSP_CAPABILITY_RECORDTYPE and
	 * MDS_CDSADIR_TP_OIDS_RECORDTYPE.
	 */
	void parseCspCapabilitiesRecord(
		MDSDictionary 				*mdsDict);
	void parseTpPolicyOidsRecord(
		MDSDictionary 				*mdsDict);

private:
	/* could be Security.framework or a loadable bundle anywhere */
	CFBundleRef		mBundle;
	char			*mPath;
	
	/* a DL session and two open DBs - one for object directory, one for 
	 * CDSA directory */
	MDSSession		&mDl;
	CSSM_DB_HANDLE 	mObjectHand;
	CSSM_DB_HANDLE 	mCdsaDirHand;
	
	char 			*mGuid;		// should this be a CFStringRef instead?
	
	// Guid/SSID defaults
	const MDS_InstallDefaults *mDefaults;
};


} // end namespace Security

#endif /* _MDS_ATTR_PARSER_H_ */
