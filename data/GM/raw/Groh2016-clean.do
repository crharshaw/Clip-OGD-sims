log using "Groh2016-clean.log", replace

use "JDEfiles/MacroinsuranceforMicroentrepreneurs.dta", clear

* Drop non-observations

drop if missing(clientid)


* Drop last observations to get an even number of units

drop in L


* Rename one covariate

rename female x0


* Median imputation for two item non-response
count if missing(x1)
replace x1 = 50 if missing(x1)


* Merge two highly correlated binary variables
cor x12 x13
replace x12 = x12 | x13


* Generate outcomes

gen o1 = admin_loanrenewal
label var o1 "Whether or not they take a loan"

gen o2 = (m_601_2 == 1)
label var o2 "Whether or not they have made a new investment in machinery or equipment"

gen o3 = m_601_2VA
label var o3 "Amount Invested in Machinery or Equipment"

gen o4 = (m_216 == 1)
label var o4 "Introduced a new product or service"

gen o5 = (m_202A >= 2 & m_202A <= 4 & m_208 == 1)
label var o5 "Started a Second Business"

gen o6 = (m_222_1 == 1 | m_222_2 == 1)
label var o6 "Hired a new worker"

for num 1/3: replace m_903_X=. if m_903_X==99997|m_903_X==99998
gen m_profits=(m_903_1+m_903_2+m_903_3)/3
replace m_profits=0 if m_102==2|(m_207==1|m_207==2|m_207==4)
gen b_profits=(b_605_1+b_605_2+b_605_3)/3
gen miss_baseprofits=b_profits==.
replace b_profits=0 if b_profits==.
sum m_profits, d
gen m_prof_cap=m_profits
replace m_prof_cap=r(p99) if m_profits>r(p99) & m_profits~=.
gen o7 = m_prof_cap
label var o7 "Monthly profits (top-coded at 99th percentile)"

for num 1/3: replace m_901_X=. if m_901_X==99997|m_901_X==99998
gen m_revenue=(m_901_1+m_901_2+m_901_3)/3
replace m_revenue=0 if m_102==2|(m_207==1|m_207==2|m_207==4)
gen b_revenue=(b_604_1+b_604_2+b_604_3)/3
gen miss_baserevenue=b_revenue==.
replace b_revenue=0 if b_revenue==.
sum m_revenue, d
gen m_rev_cap=m_revenue
replace m_rev_cap=r(p99) if m_revenue>r(p99) & m_revenue~=.
gen o8 = m_rev_cap
label var o8 "Revenue"


* Zero imputation if missing

count if missing(o1)
replace o1 = 0 if missing(o1)

count if missing(o7)
replace o7 = 0 if missing(o7)

count if missing(o8)
replace o8 = 0 if missing(o8)


* Clean up and save

keep treat x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 o1 o2 o3 o4 o5 o6 o7 o8

foreach var of varlist x0 x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 x11 x12 o1 o2 o3 o4 o5 o6 o7 o8 {
	assert !missing(`var')
}

export delimited using "Groh2016-data.csv", nolabel quote replace
