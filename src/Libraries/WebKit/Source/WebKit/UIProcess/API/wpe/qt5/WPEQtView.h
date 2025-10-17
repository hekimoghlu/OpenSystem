/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, October 14, 2021.
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
#pragma once

#include <QQmlEngine>
#include <QQuickItem>
#include <QUrl>
#include <memory>
#include <wpe/webkit.h>

class WPEQtViewBackend;
class WPEQtViewLoadRequest;

class Q_DECL_EXPORT WPEQtView : public QQuickItem {
    Q_OBJECT
    Q_DISABLE_COPY(WPEQtView)
    Q_PROPERTY(QUrl url READ url WRITE setUrl NOTIFY urlChanged)
    Q_PROPERTY(bool loading READ isLoading NOTIFY loadingChanged)
    Q_PROPERTY(int loadProgress READ loadProgress NOTIFY loadProgressChanged)
    Q_PROPERTY(QString title READ title NOTIFY titleChanged)
    Q_PROPERTY(bool canGoBack READ canGoBack NOTIFY loadingChanged)
    Q_PROPERTY(bool canGoForward READ canGoForward NOTIFY loadingChanged)
    Q_ENUMS(LoadStatus)

public:
    enum LoadStatus {
        LoadStartedStatus,
        LoadStoppedStatus,
        LoadSucceededStatus,
        LoadFailedStatus
    };

    WPEQtView(QQuickItem* parent = nullptr);
    ~WPEQtView();
    QSGNode* updatePaintNode(QSGNode*, UpdatePaintNodeData*) final;

    void triggerUpdate() { QMetaObject::invokeMethod(this, "update"); };

    QUrl url() const;
    void setUrl(const QUrl&);
    int loadProgress() const;
    QString title() const;
    bool canGoBack() const;
    bool isLoading() const;
    bool canGoForward() const;

    WebKitWebView* webView() const;

public Q_SLOTS:
    void goBack();
    void goForward();
    void reload();
    void stop();
    void loadHtml(const QString& html, const QUrl& baseUrl = QUrl());
    void runJavaScript(const QString& script, const QJSValue& callback = QJSValue());

Q_SIGNALS:
    void webViewCreated();
    void urlChanged();
    void titleChanged();
    void loadingChanged(WPEQtViewLoadRequest* loadRequest);
    void loadProgressChanged();

protected:
    bool errorOccured() const { return m_errorOccured; };
    void setErrorOccured(bool errorOccured) { m_errorOccured = errorOccured; };

    void geometryChanged(const QRectF& newGeometry, const QRectF& oldGeometry) override;

    void hoverEnterEvent(QHoverEvent*) override;
    void hoverLeaveEvent(QHoverEvent*) override;
    void hoverMoveEvent(QHoverEvent*) override;

    void mousePressEvent(QMouseEvent*) override;
    void mouseReleaseEvent(QMouseEvent*) override;
    void wheelEvent(QWheelEvent*) override;

    void keyPressEvent(QKeyEvent*) override;
    void keyReleaseEvent(QKeyEvent*) override;

    void touchEvent(QTouchEvent*) override;

private Q_SLOTS:
    void configureWindow();
    void createWebView();

private:
    static void notifyUrlChangedCallback(WPEQtView*);
    static void notifyTitleChangedCallback(WPEQtView*);
    static void notifyLoadProgressCallback(WPEQtView*);
    static void notifyLoadChangedCallback(WebKitWebView*, WebKitLoadEvent, WPEQtView*);
    static void notifyLoadFailedCallback(WebKitWebView*, WebKitLoadEvent, const gchar* failingURI, GError*, WPEQtView*);

    WebKitWebView* m_webView { nullptr };
    QUrl m_url;
    QString m_html;
    QUrl m_baseUrl;
    QSizeF m_size;
    WPEQtViewBackend* m_backend { nullptr };
    bool m_errorOccured { false };
};
