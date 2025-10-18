<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:d="http://docbook.org/ns/docbook"
  xmlns:xlink="http://www.w3.org/1999/xlink"
  xmlns="http://www.w3.org/1999/xhtml"
  exclude-result-prefixes="d xlink"
  version="1.0">

<!-- See "Customizing DocBook 5 XSL" at http://www.sagehill.net/docbookxsl/CustomDb5Xsl.html -->
<xsl:import href="http://docbook.sourceforge.net/release/xsl-ns/current/xhtml/chunk.xsl" />

<!-- For a list of available parameters, see http://docbook.sourceforge.net/release/xsl/current/doc/html/ -->
<xsl:param name="html.stylesheet" select="'../style/style.css'" />
<xsl:param name="chunker.output.encoding" select="'utf-8'" />
<xsl:param name="chunker.output.indent" select="'yes'" />
<xsl:param name="chunk.section.depth" select="0" />
<xsl:param name="suppress.navigation" select="1" />
<xsl:param name="use.id.as.filename" select="1" />
<xsl:param name="root.filename" select="'root'" />
<xsl:param name="toc.max.depth" select="0" />
<xsl:param name="css.decoration" select="0" />
<xsl:param name="generate.toc">
book nop
</xsl:param>

<xsl:template name="chunk-element-content">
  <xsl:param name="prev"/>
  <xsl:param name="next"/>
  <xsl:param name="nav.context"/>
  <xsl:param name="content">
    <xsl:apply-imports/>
  </xsl:param>

  <xsl:call-template name="user.preroot"/>

  <html>
    <xsl:call-template name="html.head"></xsl:call-template>

    <body>
      <xsl:call-template name="body.attributes"/>
      <xsl:call-template name="user.header.navigation"/>

      <xsl:call-template name="header.navigation">
        <xsl:with-param name="prev" select="$prev"/>
        <xsl:with-param name="next" select="$next"/>
        <xsl:with-param name="nav.context" select="$nav.context"/>
      </xsl:call-template>

      <xsl:call-template name="user.header.content"/>

      <div id="content-wrapper">
        <div id="sidebar">
          <xsl:apply-templates select="//d:book/d:info/*/d:formalpara" />
          <xsl:apply-templates select="//d:book/d:info/*/d:para[@xml:id='sidebar-credits']" />
        </div>
        <div id="content">
          <xsl:copy-of select="$content"/>
        </div>
      </div>

      <xsl:call-template name="user.footer.content"/>

      <xsl:call-template name="footer.navigation">
        <xsl:with-param name="prev" select="$prev"/>
        <xsl:with-param name="next" select="$next"/>
        <xsl:with-param name="nav.context" select="$nav.context"/>
      </xsl:call-template>

      <xsl:call-template name="user.footer.navigation"/>
    </body>
  </html>
  <xsl:value-of select="$chunk.append"/>
</xsl:template>

<xsl:template name="html.head">
	<head>
	<xsl:call-template name="system.head.content"/>
	<xsl:if test="$html.stylesheet != ''">
		<xsl:call-template name="output.html.stylesheets">
			<xsl:with-param name="stylesheets" select="normalize-space($html.stylesheet)"/>
		</xsl:call-template>
	</xsl:if>
	<meta name="generator" content="DocBook {$DistroTitle} V{$VERSION}"/>
	<title>
		<xsl:value-of select="d:section/d:title"/>
		- <xsl:value-of select="//d:book/d:info/d:title"/>
		- <xsl:value-of select="//d:book/d:info/d:subtitle"/>
	</title>

	<xsl:call-template name="user.head.content"/>
	</head>
</xsl:template>

<xsl:template name="user.header.navigation">
	<xsl:apply-templates select="//d:book/d:info" />
</xsl:template>

<!-- <xsl:template name="user.header.content"></xsl:template> -->

<xsl:template name="footer.navigation">
	<xsl:apply-templates select="//d:book/d:info/*/d:para[@xml:id='footer']" />
</xsl:template>

<xsl:template name="article.titlepage"></xsl:template>

<xsl:template name="article.titlepage.separator">
</xsl:template>

<xsl:template match="d:book/d:info">
	<div id="header">
		<div class="content">
			<h1><xsl:value-of select="d:title" /></h1>
			<p><xsl:value-of select="d:subtitle" /></p>
		</div>
		<div class="banner">
			<xsl:apply-templates select="//d:para[@xml:id='banner']/d:itemizedlist" />
		</div>
	</div>
</xsl:template>

<xsl:template match="d:listitem/d:para">
  <xsl:apply-templates />
</xsl:template>

<xsl:template match="d:book/d:info/*/d:formalpara">
	<div id="{@xml:id}">
		<h2><xsl:value-of select="d:title" /></h2>
		<xsl:apply-templates select="d:para" />
  </div>
</xsl:template>

<xsl:template match="d:book/d:info/*/d:para[@xml:id='sidebar-credits']">
	<div id="{@xml:id}">
    <xsl:apply-templates />
  </div>
</xsl:template>

<xsl:template match="d:book/d:info/*/d:para[@xml:id='footer']">
	<div id="footer">
		<xsl:apply-templates />
	</div>
</xsl:template>

<xsl:template match="d:note">
	<div class="highlight">
		<xsl:apply-templates />
	</div>
</xsl:template>

<xsl:template match="d:application">
	<strong><xsl:value-of select="." /></strong>
</xsl:template>

<xsl:template match="d:command">
  <pre class="command"><xsl:value-of select="." /></pre>
</xsl:template>

<xsl:template match="d:informalfigure">
  <a href="{@xlink:href}"><xsl:apply-templates select="d:mediaobject/d:imageobject" /></a>
</xsl:template>
</xsl:stylesheet>
