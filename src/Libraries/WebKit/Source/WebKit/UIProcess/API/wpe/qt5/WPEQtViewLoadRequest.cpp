/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, March 25, 2024.
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
#include "config.h"
#include "WPEQtViewLoadRequest.h"

#include "WPEQtView.h"
#include "WPEQtViewLoadRequestPrivate.h"

/*!
  \qmltype WPEViewLoadRequest
  \instantiates WPEQtViewLoadRequest
  \inqmlmodule org.wpewebkit.qtwpe

  \brief A utility type for \l {WPEView}'s \l {WPEView::}{loadingChanged()} signal.

  The WPEViewLoadRequest type contains load status information for the requested URL.

  \sa {WPEView::loadingChanged()}{WPEView.loadingChanged()}
*/
WPEQtViewLoadRequest::WPEQtViewLoadRequest(const WPEQtViewLoadRequestPrivate& d)
    : d_ptr(new WPEQtViewLoadRequestPrivate(d))
{

}

WPEQtViewLoadRequest::~WPEQtViewLoadRequest()
{

}

/*!
  \qmlproperty url WPEView::WPEViewLoadRequest::url
  \readonly

  The URL of the load request.
*/
QUrl WPEQtViewLoadRequest::url() const
{
    Q_D(const WPEQtViewLoadRequest);
    return d->m_url;
}

/*!
  \qmlproperty enumeration WPEViewLoadRequest::status
  \readonly

  This enumeration represents the load status of a web page load request.

  \value WPEView.LoadStartedStatus The page is currently loading.
  \value WPEView.LoadStoppedStatus The page loading was interrupted.
  \value WPEView.LoadSucceededStatus The page was loaded successfully.
  \value WPEView.LoadFailedStatus The page could not be loaded.

  \sa {WPEView::loadingChanged()}{WPEView.loadingChanged}
*/
WPEQtView::LoadStatus WPEQtViewLoadRequest::status() const
{
    Q_D(const WPEQtViewLoadRequest);
    return d->m_status;
}

/*!
  \qmlproperty string WPEView::WPEViewLoadRequest::errorString
  \readonly

  Holds the error message if the load request failed.
*/
QString WPEQtViewLoadRequest::errorString() const
{
    Q_D(const WPEQtViewLoadRequest);
    return d->m_errorString;
}
