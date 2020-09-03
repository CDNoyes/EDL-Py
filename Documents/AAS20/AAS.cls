%%
%% This is file `AAS.cls',
%% Author:	Hanspeter Schaub
%% Date:		March 8, 2003
%% Update:	March 1, 2010
%% Version:	2.0.5
%% Purpose: 	Provides a modern LaTeX style file for the
%%			American Astronautical Society (AAS) papers
%%
%%
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{AAS}
              [2010/03/01 v2.0.5
 Standard LaTeX document class]
\newif\if@submit
\newif\if@paper
\newif\if@cover
\newif\if@nopagenum
\newif\if@preprint
\@submitfalse
\@papertrue
\@coverfalse
\@nopagenumfalse
\@preprintfalse
\DeclareOption{paper}{\@papertrue\@submitfalse}
\DeclareOption{submit}{\@submittrue\@paperfalse}
\DeclareOption{cover}{\@covertrue}
\DeclareOption{nopagenum}{\@nopagenumtrue}
\DeclareOption{preprint}{\@preprinttrue}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}%
                \typeout{NOTE: Passing ``\CurrentOption" option on to the
                         standard LaTeX article class}}
\ProcessOptions
\typeout{}
\if@submit%
  \typeout{NOTE: AAS journal submission mode
          - all other AAS options ignored}
\else% paper
  \if@paper%
    \typeout{TYPESETTING in AAS conference PAPER format ...}
  \fi
\fi
\typeout{}

\RequirePackage{graphicx}
\RequirePackage{setspace}
%\RequirePackage{overcite}
\RequirePackage{fancyhdr}
\RequirePackage{tabls}
\RequirePackage{indentfirst}
\RequirePackage{times}
\RequirePackage{color}






\newcommand\@ptsize{}
\newcommand*{\SubmitName}[1]%
            {\def\AA@submitname{#1}}
\newcommand*{\PaperNumber}[1]%
            {\def\AA@papernumber{#1}}
\newcommand*{\JournalName}[1]%
            {\def\AA@journalname{#1}}
\newcommand*{\JournalIssue}[1]%
            {\def\AA@journalissue{#1}}
\newcommand*{\JournalAuthors}[1]%
            {\def\AA@journalauthors{#1}}
\newcommand*{\JournalTitle}[1]%
            {\def\AA@journaltitle{#1}}
\SubmitName{}
\PaperNumber{}
\JournalName{}


\newcommand{\Conference}[1]{\def\AA@conference{#1}}
\Conference{}
\newcommand{\CoverFigure}[1]{\def\AA@coverfigure{#1}}
\CoverFigure{}
\newcommand{\ConferenceLocation}[1]{\def\AA@conferencelocation{#1}}
\ConferenceLocation{}
\newcommand{\ConferenceDate}[1]{\def\AA@conferencedate{#1}}
\ConferenceDate{}



\newif\if@restonecol
\newif\if@titlepage
\@titlepagefalse
\if@compatibility\else
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\fi





\if@compatibility
  \renewcommand\@ptsize{0}
\else
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\fi
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\if@compatibility\else
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\fi
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\if@compatibility\else
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\fi
\DeclareOption{titlepage}{\@titlepagetrue}
\if@compatibility\else
\DeclareOption{notitlepage}{\@titlepagefalse}
\fi
\if@compatibility\else
\DeclareOption{onecolumn}{\@twocolumnfalse}
\fi
\DeclareOption{twocolumn}{\@twocolumnfalse}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{%
  \AtEndOfPackage{%
   \renewcommand\@openbib@code{%
      \advance\leftmargin\bibindent
      \itemindent -\bibindent
      \listparindent \itemindent
      \parsep \z@
      }%
   \renewcommand\newblock{\par}}%
}
\ExecuteOptions{letterpaper,10pt,twoside,onecolumn,final}
\ProcessOptions
\input{size1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}



\if@submit
  \setlength{\textheight}{9in}
    \setlength{\textwidth}{6.5in}
    \setlength{\columnsep}{0.25in}
    \setlength{\footskip}{30pt}
    \setlength{\parindent}{1.0em}
    \setlength{\topmargin}{-0.4in}
    \setlength{\oddsidemargin}{0.in}
    \setlength{\evensidemargin}{0.in}
    \setlength{\headsep}{0.1in}
    \setlength{\headheight}{0.05in}
  \setlength{\evensidemargin}{\oddsidemargin}
  \AtBeginDocument{\doublespacing}% turn on `doublespacing'
\else% paper
 \if@paper
	  \setlength{\textheight}{8.75in}
	    \setlength{\textwidth}{6in}
	    \setlength{\columnsep}{0.25in}
	    \setlength{\footskip}{0.9in}
	    \setlength{\parindent}{1.0em}
	    \setlength{\topmargin}{-0.45in}
	    \setlength{\oddsidemargin}{0.25in}
	    \setlength{\evensidemargin}{0.25in}
	    \setlength{\headsep}{0.1in}
	    \setlength{\headheight}{0.05in}
	  \setlength{\evensidemargin}{\oddsidemargin}
  \fi
\fi





%
%   declare the headers and footers
%

\if@submit
  	\fancyhead{}
	\fancyfoot{}
	\fancyfoot[C]{\thepage}
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
\else% paper, or note
  \if@paper
    	\fancyhead{}
	\fancyfoot{}
	\fancyfoot[C]{\thepage}
	\renewcommand{\headrulewidth}{0pt}
	\renewcommand{\footrulewidth}{0pt}
  \fi
\fi

\if@nopagenum
	\fancyfoot[C]{}
\fi





\if@paper
	\setlength\parskip{5\p@ \@plus \p@}
\fi
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\if@twoside
  \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
    \def\sectionmark##1{%
      \markboth {\MakeUppercase{%
        \ifnum \c@secnumdepth >\z@
          \thesection\quad
        \fi
        ##1}}{}}%
    \def\subsectionmark##1{%
      \markright {%
        \ifnum \c@secnumdepth >\@ne
          \thesubsection\quad
        \fi
        ##1}}}
\else
  \def\ps@headings{%
    \let\@oddfoot\@empty
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\markboth
    \def\sectionmark##1{%
      \markright {\MakeUppercase{%
        \ifnum \c@secnumdepth >\m@ne
          \thesection\quad
        \fi
        ##1}}}}
\fi
\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\thepage\hfil\slshape\leftmark}%
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\@gobbletwo
    \let\sectionmark\@gobble
    \let\subsectionmark\@gobble
    }


\if@submit%
  \newcommand{\makecover}{}
\else% paper, or note
  \newcommand{\makecover}{%
  \begin{titlepage}
  \enlargethispage{1in}
  \setcounter{page}{0}
  \renewcommand\thanks[1]{}% locally kill the \thanks{} command
  \setlength{\unitlength}{0.1in}% unit of measure for the picture
  \begin{picture}(75,90)(5,5)%  start a 7-1/2'' x 9'' picture:
  \if@paper%
    \linethickness{0.5pt}
    \put(30,78){\includegraphics[]{AIAALogo}}
    \put(0,78){\includegraphics[]{AASLogo}}
     \put(48,95)% paper number, title, author/location
        {\makebox(0,0)[tl]{\parbox{2in}
            {\sffamily\bfseries
          \LARGE AAS~\AA@papernumber% paper number
          }}}
    \put(-2,75)% paper number, title, author/location
        {\makebox(0,0)[tl]{\parbox{6.5in}
            {\begin{center}\bfseries\sffamily\huge\MakeUppercase \@title%                title
          \\[15pt]
          \LARGE\@author\end{center}}
          }}%   author/location
    \ifx\AA@coverfigure\@empty% optional figure
      \relax
    \else
      \put(30,25){\makebox(0,0)[b]{%
          \includegraphics[]%
          {\AA@coverfigure}}}%
    \fi
    \put(30,10){\makebox(0,0)[b]{\parbox{6in}%  meeting, location, and date
        {\centering\normalfont\Huge\bfseries%
        \AA@conference}}}
    \put(15,7){\makebox(0,0)[b]{\parbox{3in}%  meeting, location, and date
        {\centering\normalfont\Large\bfseries
        \AA@conferencelocation}}}
    \put(45,7){\makebox(0,0)[b]{\parbox{3in}%  meeting, location, and date
        {\centering\normalfont\Large\bfseries
        \AA@conferencedate}}}
    \put(30,4){\makebox(0,0)[b]{\parbox{6.5in}%  meeting, location, and date
        {\centering\normalfont\large\bfseries
        AAS Publications Office, P.O. Box 28130, San Diego, CA 92198}}}
  \fi
  \end{picture}
  \end{titlepage}}
\fi

\renewcommand\thefootnote{\@fnsymbol\c@footnote}



\newcommand\maketitle{
	\if@cover
    \makecover
  \else
    \relax
  \fi
	\par
  \begingroup
    \renewcommand\thefootnote{\@fnsymbol\c@footnote}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
     \long\def\@makefntext##1{\parindent 1em\noindent  
            \hb@xt@0.0em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
	\if@nopagenum
		\thispagestyle{empty}\@thanks
	\else
		\thispagestyle{plain}\@thanks
	\fi
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}

\if@paper 
\def\@maketitle{%
  \newpage
  \begin{center}%
  \let \footnote \thanks
    {
    ~\vskip -0.15in
    \hfill
            \if@preprint
            {\fontsize{14}{14}\selectfont \sffamily\bfseries
           (Preprint) AAS~\AA@papernumber}
           \else
            {\fontsize{14}{14}\selectfont \sffamily\bfseries
            AAS~\AA@papernumber}
           \fi
          \\[25pt]
    \vskip 0.4in%
          \fontsize{14}{16}\selectfont \sffamily\bfseries\MakeUppercase \@title \par}%
    \vskip 0.2in
    {\fontsize{12}{14}\selectfont
      \lineskip 1.25em%
      %\begin{tabular}[t]{c}%
        \bfseries\textsf{\@author}
      %\end{tabular}
      \par
      }%
  \end{center}%
 %\par
   }
\else 
    % submit
   \def\@maketitle{%
  \newpage
  \begin{center}%
  \let \footnote \thanks
    {
    \hfill
     \\[25pt]
          \Large\sffamily\bfseries \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      %\begin{tabular}[t]{c}%
        \bfseries\textsf{\@author}
      %\end{tabular}
      \par}%
  \end{center}%
  \par
   }
\fi          
 
\setcounter{secnumdepth}{3}
\newcounter {part}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand \thepart {\@Roman\c@part}
\renewcommand \thesection {\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection .\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}
\newcommand\part{%
   \if@noskipsec \leavevmode \fi
   \par
   \addvspace{4ex}%
   \@afterindentfalse
   \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >\m@ne
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \ifnum \c@secnumdepth >\m@ne
       \Large\bfseries \partname\nobreakspace\thepart
       \par\nobreak
     \fi
     \huge \bfseries #2%
     \markboth{}{}\par}%
    \nobreak
    \vskip 3ex
    \@afterheading}
\def\@spart#1{%
    {\parindent \z@ \raggedright
     \interlinepenalty \@M
     \normalfont
     \huge \bfseries #1\par}%
     \nobreak
     \vskip 3ex
     \@afterheading}
 
 \setcounter{secnumdepth}{0}



\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-1.5ex \@plus -1ex \@minus -.2ex}%
                                   {0.5ex \@plus.2ex}%
                                   {\normalsize\bfseries\MakeUppercase}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-1.25ex\@plus -1ex \@minus -.2ex}%
                                     {0.5ex \@plus .2ex}%
                                     {\bfseries}}

\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {0ex\@plus 0ex \@minus 0ex}%
                                     {-2ex \@plus 1ex}%
                                     {\indent\normalsize\itshape }}


     

\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\normalfont\normalsize\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\normalfont\normalsize\bfseries}}
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\leftmargin  \leftmargini
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
%\if@titlepage
%  \newenvironment{abstract}{%
%      \titlepage
%      \null\vfil
%      \@beginparpenalty\@lowpenalty
%      \begin{center}%
%        \bfseries \abstractname
%        \@endparpenalty\@M
%      \end{center}}%
%     {\par\vfil\null\endtitlepage}
%\else
  \newenvironment{abstract}{%
      \if@submit
       \section*{\abstractname}%
     \else
	\if@paper
	        \small
	        \quotation
	          \baselineskip=10pt
		  \noindent
		  \hsize=4.5in\leftskip=.3625in\parskip=4pt \rightskip=.3625in
	  \fi
      \fi
	  }
      {\if@twocolumn\else\endquotation\fi}
%\fi
\newenvironment{verse}
               {\let\\\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item\relax}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\if@compatibility
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\z@
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
    }
\else
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \setcounter{page}\@ne
    }%
    {\if@restonecol\twocolumn \else \newpage \fi
     \if@twoside\else
        \setcounter{page}\@ne
     \fi
    }
\fi
\newcommand\appendix{\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \gdef\thesection{\@Alph\c@section}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\renewcommand \theequation {\@arabic\c@equation}
\newcounter{figure}
\renewcommand \thefigure {\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\if@paper
	\def\fnum@figure{ \hsize=5.5in\leftskip=0.5in\parskip=0pt \rightskip=0.0in \normalfont\fontsize{10}{10}\selectfont	\bfseries \figurename\nobreakspace\thefigure}
\else
	\def\fnum@figure{ \leftskip=0.5in\parskip=0pt \rightskip=0.0in \bfseries\sffamily\footnotesize	\figurename\nobreakspace\thefigure}
\fi
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\if@paper
	\def\fnum@table{\bfseries\tablename\nobreakspace\thetable}
\else
	\def\fnum@table{\bfseries\tablename\nobreakspace\thetable}
\fi
\newenvironment{table}
               {
               \@float{table}
               } 
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{5\p@}
\setlength\belowcaptionskip{5\p@}
\if@paper
	  \long\def\@makecaption#1#2{%
	  \vskip\abovecaptionskip
	  \sbox\@tempboxa{\normalfont\fontsize{10}{10}\selectfont #1. \nobreakspace#2}%
	  \ifdim \wd\@tempboxa >\hsize
	  \normalfont\fontsize{10}{10}\selectfont #1. \nobreakspace#2%
%	     #1 \nobreakspace \nobreakspace #2\par
	  \else
	    \global \@minipagefalse
	    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
	  \fi
	  \vskip\belowcaptionskip}
\else
	\long\def\@makecaption#1#2{%
	  \vskip\abovecaptionskip
	  \sbox\@tempboxa{\bfseries\sffamily\small #1. #2}%
	  \ifdim \wd\@tempboxa >\hsize
	     #1. #2\par
	  \else
	    \global \@minipagefalse
	    \hb@xt@\hsize{\hfil\box\@tempboxa\hfil}%
	  \fi
	  \vskip\belowcaptionskip}
\fi
  
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{3}
\newcommand\tableofcontents{%
    \section*{\contentsname
        \@mkboth{%
           \MakeUppercase\contentsname}{\MakeUppercase\contentsname}}%
    \@starttoc{toc}%
    }
\newcommand*\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty\@secpenalty
    \addvspace{2.25em \@plus\p@}%
    \setlength\@tempdima{3em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hb@xt@\@pnumwidth{\hss #2}}\par
       \nobreak
       \if@compatibility
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}%
      \fi
    \endgroup
  \fi}
\newcommand*\l@section[2]{%
  \ifnum \c@tocdepth >\z@
    \addpenalty\@secpenalty
    \addvspace{1.0em \@plus\p@}%
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hb@xt@\@pnumwidth{\hss #2}\par
    \endgroup
  \fi}
\newcommand*\l@subsection{\@dottedtocline{2}{1.5em}{2.3em}}
\newcommand*\l@subsubsection{\@dottedtocline{3}{3.8em}{3.2em}}
\newcommand*\l@paragraph{\@dottedtocline{4}{7.0em}{4.1em}}
\newcommand*\l@subparagraph{\@dottedtocline{5}{10em}{5em}}
\newcommand\listoffigures{%
    \section*{\listfigurename}%
      \@mkboth{\MakeUppercase\listfigurename}%
              {\MakeUppercase\listfigurename}%
    \@starttoc{lof}%
    }
\newcommand*\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \section*{\listtablename}%
      \@mkboth{%
          \MakeUppercase\listtablename}%
         {\MakeUppercase\listtablename}%
    \@starttoc{lot}%
    }
\let\l@table\l@figure
\newdimen\bibindent
\setlength\bibindent{1.5em}
\newenvironment{thebibliography}[1]
     {\section*{\refname}%
      \@mkboth{\MakeUppercase\refname}{\MakeUppercase\refname}%
	\fontsize{10}{10}\selectfont
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
      	     \setlength{\itemsep}{-0.5ex plus0.0ex}
            \advance\leftmargin\labelsep
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\section*{\indexname}]%
                \@mkboth{\MakeUppercase\indexname}%
                        {\MakeUppercase\indexname}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\refname{References}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\appendixname{Appendix}
\newcommand\abstractname{Abstract}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{fancy}
\pagenumbering{arabic}
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi
\endinput


