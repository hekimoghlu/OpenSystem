/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, May 14, 2023.
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
#ifndef _ConfigPropertySheet_h
#define _ConfigPropertySheet_h

#include "stdafx.h"
#include "ServicesPage.h"
#include "RegistrationPage.h"
#include "BrowsingPage.h"

#include <RegNames.h>
#include <dns_sd.h>
#include <list>


//---------------------------------------------------------------------------------------------------------------------------
//	CConfigPropertySheet
//---------------------------------------------------------------------------------------------------------------------------

class CConfigPropertySheet : public CPropertySheet
{
public:

	CConfigPropertySheet();
	virtual ~CConfigPropertySheet();

	typedef std::list<CString> StringList;

	StringList	m_browseDomains;

protected:

	CServicesPage		m_firstPage;
	CRegistrationPage	m_secondPage;
	CBrowsingPage		m_thirdPage;

	//{{AFX_VIRTUAL(CConfigPropertySheet)
	//}}AFX_VIRTUAL

	DECLARE_DYNCREATE(CConfigPropertySheet)

	//{{AFX_MSG(CConfigPropertySheet)
	//}}AFX_MSG
	DECLARE_MESSAGE_MAP()

	afx_msg BOOL	OnInitDialog();
	afx_msg BOOL	OnCommand( WPARAM wParam, LPARAM lParam );
	afx_msg LRESULT	OnDataReady( WPARAM inWParam, LPARAM inLParam );
	afx_msg LRESULT	OnRegistryChanged( WPARAM inWParam, LPARAM inLParam );
	void			OnEndDialog();

private:

	OSStatus
	SetupBrowsing();

	OSStatus
	TearDownBrowsing();

	OSStatus
	DecodeDomainName( const char * raw, CString & decoded );

	static void DNSSD_API
	BrowseDomainsReply
				(
				DNSServiceRef			sdRef,
				DNSServiceFlags			flags,
				uint32_t				interfaceIndex,
				DNSServiceErrorType		errorCode,
				const char			*	replyDomain,
				void				*	context
				);

	// This thread will watch for registry changes

	static unsigned WINAPI
	WatchRegistry
				(
				LPVOID inParam
				);

	HKEY				m_statusKey;
	HANDLE				m_thread;
	HANDLE				m_threadExited;
	DNSServiceRef		m_browseDomainsRef;
	CRITICAL_SECTION	m_lock;
};


#endif
