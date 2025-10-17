/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 6, 2025.
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

// VersionInfo.h: interface for the CVersionInfo class.
//
//////////////////////////////////////////////////////////////////////

#if !defined(AFX_VERSIONINFO_H__F82E9FF3_5298_11D4_AB87_00C04F789BA0__INCLUDED_)
#define AFX_VERSIONINFO_H__F82E9FF3_5298_11D4_AB87_00C04F789BA0__INCLUDED_

#if _MSC_VER > 1000
#pragma once
#endif // _MSC_VER > 1000

class CVersionInfo  
{
public:
	CVersionInfo(CString filename);
	virtual ~CVersionInfo();
	BOOL IsValid() {return m_isValid;}
	DWORD GetStatus() {return m_status;}

	BOOL CopyFileCheckVersion(CVersionInfo &originalFile);
	BOOL CopyFileNoVersion(CVersionInfo &originalFile);

	const CString &GetFilename() {return m_filename;}

	// Extract the elements of the file's string info block
	CString GetFileVersionString();
	CString GetProductVersionString();
	CString GetComments();
	CString GetFileDescription();
	CString GetInternalName();
	CString GetLegalCopyright();
	CString GetLegalTrademarks();
	CString GetOriginalFileName();
	CString GetProductName();
	CString GetSpecialBuildString();
	CString GetPrivateBuildString();
	CString GetCompanyName();


	// Extract the elements of the file's VS_FIXEDFILEINFO block
	_int64 GetFileVersion();
	_int64 GetProductVersion();
	_int64 GetFileDate();

	DWORD GetFileFlagMask();
	DWORD GetFileFlags();
	DWORD GetFileOS();
	DWORD GetFileType();
	DWORD GetFileSubType();

private:
	CString m_filename;
	BOOL m_isValid;
	LPVOID m_versionInfo;
	VS_FIXEDFILEINFO *m_fixedInfo;
	DWORD m_codePage;
	DWORD m_status;

	CString QueryStringValue(CString value);
};

#endif // !defined(AFX_VERSIONINFO_H__F82E9FF3_5298_11D4_AB87_00C04F789BA0__INCLUDED_)
