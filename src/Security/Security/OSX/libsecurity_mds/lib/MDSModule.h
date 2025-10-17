/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, March 28, 2023.
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
#ifndef _MDSMODULE_H_
#define _MDSMODULE_H_  1

#include <security_filedb/AppleDatabase.h>
#include <security_utilities/globalizer.h>
#include <security_utilities/timeflow.h>
#include <security_utilities/threading.h>
#include <sys/param.h>

namespace Security
{

class MDSModule
{
public:
    static MDSModule &get ();

    MDSModule ();
    ~MDSModule ();

    DatabaseManager 		&databaseManager () { return mDatabaseManager; }
	void					lastScanIsNow();
	double					timeSinceLastScan();
	void					getDbPath(char *path);
	void					setDbPath(const char *path);
	
	bool					serverMode() const	{ return mServerMode; }
	void					setServerMode();
	
private:
    static ModuleNexus<MDSModule> mModuleNexus;

    AppleDatabaseManager 	mDatabaseManager;
	
	/*
	 * Manipulated by MDSSession objects when they hold the system-wide per-user
	 * MDS file lock. mDbPath readable any time; it's protected process-wide
	 * by mDbPathLock.
	 */
	char 					mDbPath[MAXPATHLEN + 1];
	Time::Absolute			mLastScanTime;
	Mutex					mDbPathLock;
	bool					mServerMode;
};

} // end namespace Security

#endif // _MDSMODULE_H_
