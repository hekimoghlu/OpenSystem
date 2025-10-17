/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 15, 2025.
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
#ifndef	__LOGIN_DIALOG__
#define	__LOGIN_DIALOG__

#pragma once

#include	"Resource.h"

//===========================================================================================================================
//	LoginDialog
//===========================================================================================================================

class	LoginDialog : public CDialog
{
	protected:
	
		CString		mUsername;
		CString		mPassword;
		
	public:
		
		enum { IDD = IDD_LOGIN };
		
		LoginDialog( CWnd *inParent = NULL );
		
		virtual BOOL	GetLogin( CString &outUsername, CString &outPassword );
	
	protected:

		virtual BOOL	OnInitDialog( void );
		virtual void	DoDataExchange( CDataExchange *inDX );
		virtual void	OnOK( void );
		
		DECLARE_MESSAGE_MAP()
};

#endif	// __LOGIN_DIALOG__
