SELECT c.Email as "C Email"
		,c.Active_Service_Partner__c as "C Active Service Partner?"
		,c.Active_Subscription__c as "C Active Subscription?"
		,c.CreatedDate as "C Created Date"
        ,ocr.isPrimary
		,o.Account_Subscription_Year__c as "Account Subscription Year"
		,o.AccountId as "Opportunity Account Id"
		,o.Amount__USD as "Opportunity Amount USD" 
		,o.Closed_Lost_Date__c as "Opportunity Closed Lost Date"
		,o.Closed_Won_Date__c as "Opportunity Closed Won Date"
		,o.CloseDate as "Close Date"
                ,o.CreatedDate as "Opportunity Create Date"
		,o.Details_of_Lost_Opportunity__c as "Details of Lost Opportunity"
		,o.Id_ as "Opportunity Id"
		,o.IsWon as "Is Won?"
		,o.Liferay_Version__c as "Liferay Version"
		,o.Name as "Opportunity Name"
		,o.Opportunity_Age_to_Close__c as "Opportunity Age to Close"
		,o.AccountAddress_Shipping_Address_Region__c as "Account Address Shipping Address Region"
		,o.AccountAddress_Billing_Address_Region__c as "Account Address Billing Address Region"
		,o.Product_Name__c as "Product Name"
		,o.Renewal_Dates__c as "Renewal Dates"
		,o.Sold_By__c as "Sold By"
		,o.StageName as "Stage Name"
		,o.Term_Length__c as "Term Length"
		,o.Term_Type__c as "Term Type"
		,o.Type_ as "Opportunity Type"
		,CONCAT("https:#na5.salesforce.com/" ,o.Id_) as "Opportunity URL"
		,CASE
			WHEN o.NA_Sales_Territory__c = "NorthEast" THEN "East"
			WHEN o.NA_Sales_Territory__c = "MidWest" THEN "West"
			WHEN o.NA_Sales_Territory__c = "CA Central" THEN "Central"
			WHEN o.NA_Sales_Territory__c = "CA East" THEN "East"
			WHEN o.NA_Sales_Territory__c = "CA West" THEN "West"
		ELSE o.NA_Sales_Territory__c
		END AS "SFDC Opportunity NA Sales Territory"
		,CASE 
			WHEN o.AccountAddress_Shipping_Address_Region__c IS NOT NULL
				THEN o.AccountAddress_Shipping_Address_Region__c
			WHEN o.AccountAddress_Shipping_Address_Region__c IS NULL AND o.AccountAddress_Billing_Address_Region__c IS NOT NULL
				THEN o.AccountAddress_Billing_Address_Region__c
		ELSE "Other"
		END as "SFDC Opportunity Hierarchy Region"
		,CASE 
			WHEN o.Product_Family__c = "C" THEN "Consulting"
			WHEN o.Product_Family__c = "CT" THEN "Consulting | Training"
			WHEN o.Product_Family__c = "O" THEN "Other"
			WHEN o.Product_Family__c = "P" THEN "Partnership"
			WHEN o.Product_Family__c = "S" THEN "Subscription"
			WHEN o.Product_Family__c = "SC" THEN "Subscription | Consulting"
			WHEN o.Product_Family__c = "SCT" THEN "Subscription | Consulting | Training"
			WHEN o.Product_Family__c =  "SO" THEN "Subscription | Other"
			WHEN o.Product_Family__c = "ST" THEN "Subscription | Training"
			WHEN o.Product_Family__c = "T" THEN "Training"
            ELSE o.Product_Family__c 
            END AS "Product Family"
            
FROM Contact_ as c
LEFT JOIN OpportunityContactRole as ocr
ON c.Id_ = ocr.ContactId    
LEFT JOIN Opportunity as o
ON ocr.OpportunityId  = o.Id_
WHERE o.Name not like '%liferay%'
AND o.Name not like '%deathray%'
