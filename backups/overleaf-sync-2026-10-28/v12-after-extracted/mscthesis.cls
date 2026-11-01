\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mscthesis}[2025/12/01 v1.1 UvA Software Engineering MSc Thesis]

\newcommand{\MScThesisClassDefaults}{%
  \PassOptionsToClass{a4paper,twoside,onecolumn,openright,10pt}{memoir}%
}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{memoir}}

\AtBeginDocument{%
  \ifx\ProcessedOptions\relax
    \MScThesisClassDefaults
  \fi
}
\ProcessOptions

\LoadClass{memoir}

% ===================
% = Font Selections =
% ===================
\RequirePackage{fontspec}
\setmainfont{Crimson Pro}[Numbers={OldStyle, Proportional}]
\setsansfont{PT Sans}[Scale=MatchLowercase]
\setmonofont{Source Code Pro}[Scale=MatchLowercase]

% ====================
% = Chapter headings =
% ====================
\RequirePackage{graphicx}
\chapterstyle{madsen}
\pagestyle{companion}
\setsecnumdepth{subsection}

\newcommand{\MONTH}{%
  \ifcase\the\month
  \or January% 1
  \or February% 2
  \or March% 3
  \or April% 4
  \or May% 5
  \or June% 6
  \or July% 7
  \or August% 8
  \or September% 9
  \or October% 10
  \or November% 11
  \or December% 12
  \fi}

%---------------------------------------------------------------------%
%                     Define new variables                            %    
%---------------------------------------------------------------------%

\NewDocumentCommand{\subtitle}{m}{\gdef\@subtitle{#1}}
\NewDocumentCommand{\university}{m}{\gdef\@university{#1}}

% Thesis Committee members
\NewDocumentCommand{\examiner}{mm}{\gdef\@examinername{#1}\gdef\@examineraffiliation{#2}}
\NewDocumentCommand{\reviewer}{mm}{\gdef\@reviewername{#1}\gdef\@revieweraffilication{#2}}
\NewDocumentCommand{\dailysupervisor}{mm}{\gdef\@dailysupervisorname{#1}\gdef\@dailysupervisoraffilication{#2}}
\NewDocumentCommand{\externalsupervisor}{mm}{\gdef\@externalsupervisorname{#1}\gdef\@externalsupervisoraffiliation{#2}}
\NewDocumentCommand{\othermembers}{m}{\gdef\@othermemberslist{#1}}

% Colophon
\newif\if@hascolophon 
\gdef\@colophon{\noindent{}Copyright \textcopyright{} \theauthor{}, \the\year{}.}

\NewDocumentCommand{\colophon}{m}{%
  \@hascolophontrue
  \gdef\@colophon{#1}%
}

% Host organisation
\newif\if@hashostorganisation
\gdef\@hostorganisation{}

\NewDocumentCommand{\hostorganisation}{mm}{%
  \@hashostorganisationtrue
  \gdef\@hostorganisationlogo{#1}\gdef\@hostorganisation{#2}
}

% Cover picture
\newif\if@hascoverpicture
\gdef\@coverpicture{}

\NewDocumentCommand{\coverpicture}{m}{%
  \@hascoverpicturetrue
  \gdef\@coverpicture{#1}%
}

\university{%
{\includegraphics[height=2cm,keepaspectratio]{figures/msc-se-logo-small-white.png}}\\[-4pt]
\textsf{\Large{Master Software Engineering}} \\
Informatics Institute \\
FNWI, University of Amsterdam \\
Amsterdam, The Netherlands
}

% =====================
% = Formal Title Page =
% =====================
\def\@REPTYPE{MASTER OF SCIENCE}
\def\@MAJOR{SOFTWARE ENGINEERING}

\def\makeformaltitlepage{%
	\begingroup\newpage
    	\thispagestyle{empty}
    	\let\footnotesize\small \let\footnoterule\relax
    	\noindent\rule{\textwidth}{1.5pt}
    	\vskip 1em
    	\begin{center}
    		{\Huge \@title \par}
    		\vskip 1em
    		\rule{\textwidth}{1.5pt} \par \nobreak
    		\vskip 2em
    		{\large THESIS}
    		\vskip 2em
    		{\large submitted in partial fulfilment of the \\
    		requirements for the degree of}
    		\vskip 2em
    		{\large \@REPTYPE}
    		\vskip 2em
    		in
    		\vskip 2em
    		{\large \@MAJOR}
    		\vskip 2em
    		by
    		\vskip 2em 
    		{\large \@author}
        \vskip 2em
        under the supervision of
        \vskip 0.5em
        {\large \@examinername{} (\@examineraffiliation{})}
        \vskip 2.5em
        \MONTH{} \the\year
    		\null\vfill
        \ifdefined\@hostorganisation
          \end{center}
          \begin{tabular}{@{}l@{}r@{}}
            \begin{minipage}[b]{.54\textwidth}\@university\end{minipage} & 
            \begin{minipage}[b]{.46\textwidth}\begin{flushright}
              % \includegraphics[height=2cm,keepaspectratio]{\@hostorganisationlogo}\\
              \includesvg[height=1.5cm,keepaspectratio]{\@hostorganisationlogo}\\
              \@hostorganisation
            \end{flushright}\end{minipage}
          \end{tabular}
        \else
          \@university
          \end{center}
        \fi
        \newpage
        \thispagestyle{empty}
        \noindent\begin{tabular}{@{}lp{.8\textwidth{}}@{}}
        \emph{Thesis Committee}\\\\
        Examiner (chair): & \@examinername{} (\@examineraffiliation{})\\
        Second Reviewer: & \@reviewername{} (\@revieweraffilication{})\\
        \ifdefined\@dailysupervisorname
        Daily Supervisor: & \@dailysupervisorname{} (\@dailysupervisoraffilication{})\\
        \fi
        \ifdefined\@externalsupervisorname
        External Supervisor: & \@externalsupervisorname{} (\@externalsupervisoraffiliation{})\\
        \fi
        \ifdefined\@othermemberslist
          Other members: & \@othermemberslist{} \\
        \fi
        \end{tabular}
        \null\vfill
        \footnotesize
        \setlength{\parindent}{0pt}
        \setlength{\parskip}{\baselineskip}
        \@colophon{}
        \clearpage
  \endgroup
}

% That's all, folks!
\endinput

% vim:ft=plaintex
