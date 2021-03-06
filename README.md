# fhnw-ds-fs2020-recommender_systems
## Task definition
A retail online shop wants to offer its customers personalized purchase recommendations for its products and simultaneously commissions several data scientists to develop a recommender system. In addition, the product shelf is also to be adjusted within the framework of the analyses, i.e. products that are very rarely purchased by customers are to be removed from the range. Due to the high procurement and storage costs, the company is prepared to reduce the product shelf by up to 80%.

The company provides a data extract with information on the purchasing behaviour of customers over several months. The data contains a sample of all customers and for these per purchase all articles including information on the product category.

The following delivery objects are agreed with the customer:

- Written concept for the planned procedure, in particular, cleaning up the product shelf and defining the success criteria
- Documentation (possibly presentation) of the implemented recommender system with explanation of the tested algorithms, the methodical approach and the gained insights (incl. Q&A)
- Comparison and assessment of the model quality of different model- and memory-based recommender systems with (a) methods of statistical off-line evaluation and (b) assessment of the personalisation of proposals
- Documentation of the methods, analyses and results of steps 2 and 3 in the form of a scientific report. 
- Reflection of the scientific procedure by means of a criteria sheet (basis delivery objects 1-4)

In a first step, the assignment consists of implementing a simple model- or memory-based recommender system "from scratch", i.e. without using packages, and evaluating the model quality "off-line". This should create a small experimental field in which different algorithms can be systematically tested and compared. The knowledge gained about the functionality and programming techniques should be used to make a well-founded selection from solutions offered in packages (see point 2 above). 

In a further step, packages will be used to make further systematic comparisons of several recommender system approaches and then to quantify and visualise the degree of personalisation of the sales recommendations made (see point 3 above). 

The findings from the challenge are to be documented in a systematic form as a scientific report (point 4 above), which is generated as a PDF from the code notebook. In addition, the scientific approach is to be reflected on the basis of a criteria sheet (item 5 above). 

## Data description

`orders`
- `order_id`: order identifier
- `user_id`: customer identifier
- `eval_set`: which evaluation set this order belongs in (see `SET` described below)
- `order_number`: the order sequence number for this user (1 = first, n = nth)
- `order_dow`: the day of the week the order was placed on
- `order_hour_of_day`: the hour of the day the order was placed on
- `days_since_prior`: days since the last order, capped at 30 (with NAs for order_number = 1)

`products`
- `product_id`: product identifier
- `product_name`: name of the product
- `aisle_id`: foreign key
- `department_id`: foreign key

`aisles`
- `aisle_id`: aisle identifier
- `aisle`: the name of the aisle

`deptartments`
- `department_id`: department identifier
- `department`: the name of the department

`order_products__SET`
- `order_id`: foreign key
- `product_id`: foreign key
- `add_to_cart_order`: order in which each product was added to cart
- `reordered`: 1 if this product has been ordered by this user in the past, 0 otherwise

where `SET` is one of the four following evaluation sets  (`eval_set` in `orders`):
- `"prior"`: orders prior to that users most recent order (~3.2m orders)
- `"train"`: training data supplied to participants (~131k orders)
- `"test"`: test data reserved for machine learning competitions (~75k orders)


## Contributers
- Roman Studer (roman.studer1@students.fhnw.ch, [Github](https://github.com/roman-studer))
- Lukas Gehrig (lukas.gehrig@students.fhnw.ch, [Github](https://github.com/Lukas113))
- Simon Luder (simon.luder@students.fhnw.ch, [Github](https://github.com/SimonLuder))

